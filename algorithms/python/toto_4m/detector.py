from collections import deque

import numpy as np
import torch
from toto2 import Toto2Model
from touchstone_py import Detector, run_cli

_MODEL_NAME = "Datadog/Toto-2.0-4m"
_CONTEXT_LENGTH = 1024
_WARMUP = 512
_STRIDE = 5
_QUANTILE_LEVELS = np.array(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32
)


class Toto(Detector):
    """Anomaly detector using the Datadog Toto-2.0 forecasting model.

    Inference runs every _STRIDE points rather than every point: each forward
    pass forecasts _STRIDE steps ahead and the resulting quantiles are consumed
    one-by-one as observations arrive.  Each observation is scored with the CRPS
    (continuous ranked probability score) against its pre-computed quantiles.
    Tight quantiles that miss the actual value score higher than wide quantiles
    that contain it.  The first 512 points are emitted as NaN during warmup.
    """

    @classmethod
    def name(cls) -> str:
        return "Toto-4m"

    def __init__(self, n_dimensions: int) -> None:
        self._n_dimensions = n_dimensions
        self._buffer: deque[list[float]] = deque(
            maxlen=_CONTEXT_LENGTH
        )  # grows to 1024
        self._pending_quantiles: np.ndarray | None = None  # (9, n_dims, _STRIDE)
        self._pending_idx: int = 0
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = Toto2Model.from_pretrained(_MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()

    def _forecast(self) -> np.ndarray:
        """Run one forward pass and return quantiles for the next _STRIDE steps.

        Returns:
            Array of shape (9, n_dimensions, _STRIDE).
        """
        buf = np.array(self._buffer, dtype=np.float32)  # (ctx, n_dims)
        ctx = len(buf)
        # The model requires context length divisible by its patch size (32). Always
        # pass _CONTEXT_LENGTH by left-padding short buffers with zeros marked unobserved.
        if ctx < _CONTEXT_LENGTH:
            pad = np.zeros((_CONTEXT_LENGTH - ctx, self._n_dimensions), dtype=np.float32)
            buf = np.concatenate([pad, buf], axis=0)
        target = (
            torch.tensor(buf.T).unsqueeze(0).to(self._device)
        )  # (1, n_dims, _CONTEXT_LENGTH)
        target_mask = torch.zeros(
            1, self._n_dimensions, _CONTEXT_LENGTH, dtype=torch.bool, device=self._device
        )
        target_mask[:, :, _CONTEXT_LENGTH - ctx :] = True
        series_ids = torch.zeros(
            1, self._n_dimensions, dtype=torch.long, device=self._device
        )

        with torch.no_grad():
            quantiles = self._model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=_STRIDE,
            )
        # quantiles shape: (9, batch, n_variates, horizon)
        return quantiles[:, 0, :, :].cpu().numpy()  # (9, n_dims, _STRIDE)

    @staticmethod
    def _crps(y: np.ndarray, quantiles: np.ndarray) -> float:
        """Compute mean CRPS across dimensions via the pinball loss approximation.

        Args:
            y: Actual observation of shape (n_dims,).
            quantiles: Predicted quantiles of shape (9, n_dims), levels 0.1–0.9.

        Returns:
            Mean CRPS score across all dimensions.
        """
        diff = y[np.newaxis, :] - quantiles  # (9, n_dims)
        tau = _QUANTILE_LEVELS[:, np.newaxis]  # (9, 1)
        loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
        return float(np.mean(loss) * 2)

    def update(self, point: list[float]) -> float:
        """Ingest a new observation and return its anomaly score.

        Returns NaN until the context buffer has been filled for the first time.

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

        if len(self._buffer) >= _WARMUP and (
            self._pending_quantiles is None or self._pending_idx >= _STRIDE
        ):
            self._pending_quantiles = self._forecast()
            self._pending_idx = 0

        return score


if __name__ == "__main__":
    run_cli(Toto)
