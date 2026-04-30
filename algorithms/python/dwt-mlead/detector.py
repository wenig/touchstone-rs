"""DWT-MLEAD univariate anomaly detector.

For each new point:
  1. Run a single-level causal Haar DWT step at each cascade level (à-trous).
  2. Each level produces a new detail coefficient.
  3. Score that coefficient against a running mean/std (EW statistics).
  4. If the z-score exceeds the 97th-percentile threshold, accumulate the excess.
"""
import math

from touchstone_py import Detector, run_cli

_INV_SQRT2 = 1.0 / math.sqrt(2)
_LEVELS = 5
_LAMBDA = 0.9       # EW forgetting factor for running mean/var
_Z_97 = 1.8808      # norm.ppf(0.97) — 97th-percentile threshold
_MIN_SAMPLES = 20   # warmup per level before scoring


def haar_step(prev: float, curr: float) -> tuple[float, float]:
    """Single-level causal Haar DWT on two consecutive samples.

    Returns (approx, detail) using the orthonormal Haar basis.
    """
    return (prev + curr) * _INV_SQRT2, (prev - curr) * _INV_SQRT2


class DwtMlead(Detector):
    """Univariate anomaly detector based on multi-level Haar DWT.

    An à-trous (undecimated) Haar DWT cascade decomposes the incoming stream
    into detail coefficients at multiple frequency scales. Each level maintains
    an exponentially-weighted running mean and variance of its detail
    coefficients. A point is flagged when the z-score at any level exceeds
    ``z_threshold`` (default: 97th percentile of N(0,1)); the returned score
    is the sum of per-level excesses above that threshold.

    Returns ``float('nan')`` until the minimum warmup samples have been seen.
    """

    @classmethod
    def name(cls) -> str:
        """Return the detector's display name."""
        return "DWT-MLEAD"

    def __init__(
        self,
        n_dimensions: int = 1,
        levels: int = _LEVELS,
        forgetting_factor: float = _LAMBDA,
        z_threshold: float = _Z_97,
    ) -> None:
        """Initialise the detector.

        Args:
            n_dimensions: Dimensionality of the input stream (only the first
                dimension is used; reserved for future multivariate extension).
            levels: Number of DWT decomposition levels.
            forgetting_factor: Exponential decay λ ∈ (0, 1) for the running
                mean/variance. Smaller values adapt faster.
            z_threshold: Minimum z-score to count as anomalous. Defaults to
                the 97th percentile of the standard normal (≈ 1.88).
        """
        self._levels = levels
        self._lam = forgetting_factor
        self._z = z_threshold

        # À-trous cascade: one pending approx sample per level
        self._prev: list[float | None] = [None] * levels

        # Per-level EW running mean, variance, weight, and sample count
        self._mean = [0.0] * levels
        self._var = [0.0] * levels
        self._weight = [0.0] * levels
        self._n = [0] * levels

    def update(self, point: list[float]) -> float:
        """Consume the next data point and return an anomaly score.

        Runs a single Haar DWT step at each cascade level, updates the
        running statistics for the resulting detail coefficient, then scores
        it. The score is the sum of (z - z_threshold) across all levels where
        the z-score exceeds the threshold. Returns ``float('nan')`` during
        the per-level warmup period.

        Args:
            point: Feature vector; only ``point[0]`` is used.

        Returns:
            Non-negative anomaly score, or ``float('nan')`` during warmup.
        """
        val = float(point[0])
        total = 0.0
        any_level_scored = False

        for ell in range(self._levels):
            prev = self._prev[ell]
            self._prev[ell] = val
            if prev is None:
                break  # cascade not yet primed at this depth

            approx, detail = haar_step(prev, val)
            val = approx  # pass approximation up to the next level

            # EW update of running mean and variance
            lam = self._lam
            self._weight[ell] = lam * self._weight[ell] + 1.0
            alpha = 1.0 / self._weight[ell]
            old_mean = self._mean[ell]
            self._mean[ell] += alpha * (detail - old_mean)
            self._var[ell] = (1.0 - alpha) * self._var[ell] + alpha * (detail - old_mean) * (detail - self._mean[ell])
            self._n[ell] += 1

            if self._n[ell] < _MIN_SAMPLES:
                continue

            std = math.sqrt(max(self._var[ell], 1e-10))
            z = abs(detail - self._mean[ell]) / std
            any_level_scored = True
            if z > self._z:
                total += z - self._z

        return total if any_level_scored else float("nan")


if __name__ == "__main__":
    run_cli(DwtMlead)
