from collections import deque

import numpy as np
import torch
from touchstone_py import Detector, run_cli

from model import TinyTimeMixerForPrediction

_MODEL_NAME = "ibm-granite/granite-timeseries-ttm-r2"
_MODEL_REVISION = "1024-96-r2"
_CONTEXT_LENGTH = 1024
_WARMUP = 512
_STRIDE = 5


class TinyTimeMixer(Detector):
    """Anomaly detector using the IBM Granite TinyTimeMixer (TTM-R2) forecasting model.

    Inference runs every _STRIDE points: each forward pass forecasts _STRIDE steps
    ahead and the resulting predictions are consumed one-by-one as observations
    arrive.  The anomaly score is the mean absolute error between the pre-computed
    prediction and the actual observation.  The first 512 points are emitted as NaN
    during the warmup period.
    """

    @classmethod
    def name(cls) -> str:
        return "TinyTimeMixer"

    def __init__(self, n_dimensions: int) -> None:
        self._n_dimensions = n_dimensions
        self._buffer: deque[list[float]] = deque(maxlen=_CONTEXT_LENGTH)
        self._pending_predictions: np.ndarray | None = None  # (_STRIDE, n_dims)
        self._pending_idx: int = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TinyTimeMixerForPrediction.from_pretrained(_MODEL_NAME, revision=_MODEL_REVISION)
        self._model.to(self._device)
        self._model.eval()

    def _forecast(self) -> np.ndarray:
        """Run one forward pass and return predictions for the next _STRIDE steps.

        Returns:
            Array of shape (_STRIDE, n_dimensions).
        """
        buf = np.array(self._buffer, dtype=np.float32)  # (context_length, n_dims)
        x = torch.tensor(buf).unsqueeze(0).to(self._device)  # (1, context_length, n_dims)
        with torch.no_grad():
            output = self._model(past_values=x)
        # prediction_outputs shape: (batch, prediction_length, n_dims)
        return output.prediction_outputs[0, :_STRIDE].cpu().numpy()  # (_STRIDE, n_dims)

    def update(self, point: list[float]) -> float:
        """Ingest a new observation and return its anomaly score.

        Returns NaN until the context buffer has been filled for the first time.

        Args:
            point: Observed values for the current time step, one per dimension.

        Returns:
            Anomaly score in [0, ∞), or NaN during the warmup period.
        """
        point_arr = np.array(point, dtype=np.float32)

        score = np.nan
        if self._pending_predictions is not None:
            score = float(np.mean(np.abs(point_arr - self._pending_predictions[self._pending_idx])))
            self._pending_idx += 1

        self._buffer.append(point)

        if len(self._buffer) >= _WARMUP and (self._pending_predictions is None or self._pending_idx >= _STRIDE):
            self._pending_predictions = self._forecast()
            self._pending_idx = 0

        return score


if __name__ == "__main__":
    run_cli(TinyTimeMixer)
