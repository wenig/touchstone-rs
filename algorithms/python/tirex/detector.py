from collections import deque

import numpy as np
from tirex import ForecastModel, load_model
from touchstone_py import Detector, run_cli

_MODEL_NAME = "NX-AI/TiRex"
_CONTEXT_LENGTH = 2016
_WARMUP = 512
_STRIDE = 5


class TiRex(Detector):
    """Anomaly detector using the NX-AI TiRex zero-shot forecasting model.

    Inference runs every _STRIDE points: each forward pass forecasts _STRIDE steps
    ahead per dimension and the resulting quantiles are consumed one-by-one as
    observations arrive.  Each observation is scored with the CRPS (continuous ranked
    probability score) against its pre-computed quantiles.  The first _WARMUP points
    are emitted as NaN during the warmup period.

    Each dimension is treated as an independent time series and batched together
    for a single forward pass, since TiRex expects (batch, context_length) input.
    """

    @classmethod
    def name(cls) -> str:
        return "TiRex"

    def __init__(self, n_dimensions: int) -> None:
        self._n_dimensions = n_dimensions
        self._buffer: deque[list[float]] = deque(maxlen=_CONTEXT_LENGTH)
        self._pending_quantiles: np.ndarray | None = None  # (num_quantiles, n_dims, _STRIDE)
        self._pending_idx: int = 0
        self._model: ForecastModel = load_model(_MODEL_NAME)
        self._quantile_levels = np.array(self._model.config.quantiles, dtype=np.float32)

    def _forecast(self) -> np.ndarray:
        """Run one forward pass and return quantiles for the next _STRIDE steps.

        Returns:
            Array of shape (num_quantiles, n_dimensions, _STRIDE).
        """
        buf = np.array(self._buffer, dtype=np.float32)  # (context_length, n_dims)
        # TiRex expects (batch, context_length); each dimension is a separate series
        context = buf.T  # (n_dims, context_length)
        quantiles, _ = self._model.forecast(context=context, prediction_length=_STRIDE, output_type="numpy")
        # quantiles shape from TiRex: (n_dims, _STRIDE, num_quantiles)
        # rearrange to (num_quantiles, n_dims, _STRIDE) to match Toto convention
        return quantiles.transpose(2, 0, 1)

    def _crps(self, y: np.ndarray, quantiles: np.ndarray) -> float:
        """Compute mean CRPS across dimensions via the pinball loss approximation.

        Args:
            y: Actual observation of shape (n_dims,).
            quantiles: Predicted quantiles of shape (num_quantiles, n_dims).

        Returns:
            Mean CRPS score across all dimensions.
        """
        diff = y[np.newaxis, :] - quantiles  # (num_quantiles, n_dims)
        tau = self._quantile_levels[:, np.newaxis]  # (num_quantiles, 1)
        loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
        return float(np.mean(loss) * 2)

    def update(self, point: list[float]) -> float:
        """Ingest a new observation and return its anomaly score.

        Returns NaN until _WARMUP points have been observed.

        Args:
            point: Observed values for the current time step, one per dimension.

        Returns:
            CRPS anomaly score in [0, ∞), or NaN during the warmup period.
        """
        point_arr = np.array(point, dtype=np.float32)

        score = np.nan
        if self._pending_quantiles is not None:
            score = self._crps(
                point_arr, self._pending_quantiles[:, :, self._pending_idx]
            )
            self._pending_idx += 1

        self._buffer.append(point)

        if len(self._buffer) >= _WARMUP and (self._pending_quantiles is None or self._pending_idx >= _STRIDE):
            self._pending_quantiles = self._forecast()
            self._pending_idx = 0

        return score


if __name__ == "__main__":
    run_cli(TiRex)
