from collections import deque

import numpy as np
import torch
from touchstone_py import Detector, run_cli

from model import TinyTimeMixerForPrediction

_MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r2"
_CONTEXT_LENGTH = 512


class TinyTimeMixer(Detector):
    """Anomaly detector using the IBM Granite TinyTimeMixer (TTM-R2) forecasting model.

    At each time step the model predicts the next observation from the preceding
    512-point context window.  The mean absolute error between that prediction
    and the actual observation is returned as the anomaly score.  The first 512
    points are emitted as NaN while the context buffer fills (warmup period).
    """

    @classmethod
    def name(cls) -> str:
        return "TinyTimeMixer"

    def __init__(self, n_dimensions: int) -> None:
        """Load the pretrained TTM-R2 checkpoint and initialise the context buffer.

        Args:
            n_dimensions: Number of channels in the time series.
        """
        self._n_dimensions = n_dimensions
        self._buffer: deque[list[float]] = deque(maxlen=_CONTEXT_LENGTH)
        self._next_prediction: np.ndarray | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TinyTimeMixerForPrediction.from_pretrained(_MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()

    def _predict_next(self) -> np.ndarray:
        """Run one forward pass over the current context and return the predicted next step.

        Returns:
            Array of shape (n_dimensions,) with the predicted values for the next time step.
        """
        buf = np.array(self._buffer, dtype=np.float32)  # (context_length, n_dims)
        x = torch.tensor(buf).unsqueeze(0).to(self._device)  # (1, context_length, n_dims)
        with torch.no_grad():
            output = self._model(past_values=x)
        return output.prediction_outputs[0, 0].cpu().numpy()  # (n_dims,)

    def update(self, point: list[float]) -> float:
        """Ingest a new observation and return its anomaly score.

        The score is the mean absolute error between the value predicted for this
        time step (from the previous context window) and the actual observation.
        Returns NaN until the context buffer has been filled for the first time.

        Args:
            point: Observed values for the current time step, one per dimension.

        Returns:
            Anomaly score in [0, ∞), or NaN during the warmup period.
        """
        point_arr = np.array(point, dtype=np.float32)

        score = np.nan
        if self._next_prediction is not None:
            score = float(np.mean(np.abs(point_arr - self._next_prediction)))

        self._buffer.append(point)

        if len(self._buffer) == _CONTEXT_LENGTH:
            self._next_prediction = self._predict_next()

        return score


if __name__ == "__main__":
    run_cli(TinyTimeMixer)
