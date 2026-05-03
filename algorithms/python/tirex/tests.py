"""Tests for TiRex scoring logic.

Scoring summary
---------------
- Buffer fills as a deque (maxlen=_CONTEXT_LENGTH=2016).
- No score is produced until len(buffer) >= _WARMUP (512); those updates return NaN.
- After point 511 is appended (buffer length hits _WARMUP), a forecast of shape
  (num_quantiles, n_dims, _STRIDE) is computed and stored as _pending_quantiles.
- From point 512 onward, each update consumes one pending step:
    score = CRPS(observation, pending_quantiles[:, :, pending_idx])
  and increments pending_idx.
- When pending_idx reaches _STRIDE (5), a fresh forecast is triggered and pending_idx
  resets to 0. So forecasts run every _STRIDE points after warmup.

CRPS maths (pinball loss approximation)
----------------------------------------
  diff = y - q                           # (num_q, n_dims)
  loss = tau*diff  if diff >= 0
         (tau-1)*diff  otherwise
  score = mean(loss) * 2

With quantile levels [0.1, …, 0.9] (mean tau = 0.5):
  all q = 0, y = v > 0  →  score = mean(tau) * v * 2 = 1.0 * v  (used in order test)
  all q = 0, y = 0       →  score = 0                             (perfect prediction)
"""

import math
import sys
import types
import unittest
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so we can import detector without downloading model
# ---------------------------------------------------------------------------

_touchstone_stub = types.ModuleType("touchstone_py")
_touchstone_stub.Detector = object
_touchstone_stub.run_cli = lambda cls: None
sys.modules.setdefault("touchstone_py", _touchstone_stub)

_tirex_stub = types.ModuleType("tirex")
_tirex_stub.ForecastModel = object
_tirex_stub.load_model = MagicMock()
sys.modules.setdefault("tirex", _tirex_stub)

import importlib

import detector as _detector_module

_WARMUP = _detector_module._WARMUP
_STRIDE = _detector_module._STRIDE
_CONTEXT_LENGTH = _detector_module._CONTEXT_LENGTH

_DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
_NUM_QUANTILES = len(_DEFAULT_QUANTILE_LEVELS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(quantiles: np.ndarray, quantile_levels=None):
    """Return a mock whose .forecast() returns (quantiles, None).

    quantiles: shape (n_dims, _STRIDE, num_quantiles) — TiRex's output format
               before the detector transposes it.
    """
    fake_model = MagicMock()
    fake_model.forecast.return_value = (quantiles, None)
    fake_model.config.quantiles = quantile_levels or _DEFAULT_QUANTILE_LEVELS
    return fake_model


def _build_detector(fake_model, n_dims=1):
    """Instantiate TiRex with a pre-built fake model."""
    _tirex_stub.load_model = MagicMock(return_value=fake_model)
    importlib.reload(_detector_module)
    return _detector_module.TiRex(n_dimensions=n_dims)


def _uniform_quantiles(n_dims, q_value=0.0):
    """All quantiles set to q_value; shape (n_dims, _STRIDE, _NUM_QUANTILES)."""
    return np.full((n_dims, _STRIDE, _NUM_QUANTILES), q_value, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWarmup(unittest.TestCase):
    def setUp(self):
        self.det = _build_detector(_make_fake_model(_uniform_quantiles(1)), n_dims=1)

    def test_all_nan_during_warmup(self):
        for i in range(_WARMUP):
            score = self.det.update([float(i)])
            self.assertTrue(math.isnan(score), f"Expected NaN at step {i}, got {score}")

    def test_first_post_warmup_score_is_not_nan(self):
        for _ in range(_WARMUP):
            self.det.update([0.0])
        score = self.det.update([0.0])
        self.assertFalse(math.isnan(score))


class TestCRPS(unittest.TestCase):
    """Verify the CRPS formula against hand-computed values."""

    def test_zero_score_when_all_quantiles_equal_observation(self):
        # q = y = 1.0 → diff = 0 → score = 0
        det = _build_detector(_make_fake_model(_uniform_quantiles(1, q_value=1.0)), n_dims=1)
        for _ in range(_WARMUP):
            det.update([0.0])
        score = det.update([1.0])
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_score_when_observation_above_all_quantiles(self):
        # q = 0, y = 1 → diff = 1 (positive) → loss = tau*1 → mean(tau)*2*1 = 0.5*2 = 1.0
        det = _build_detector(_make_fake_model(_uniform_quantiles(1, q_value=0.0)), n_dims=1)
        for _ in range(_WARMUP):
            det.update([0.0])
        score = det.update([1.0])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_score_when_observation_below_all_quantiles(self):
        # q = 1, y = 0 → diff = -1 (negative) → loss = (tau-1)*(-1) = 1-tau
        # mean(1-tau) = 1 - mean(tau) = 1 - 0.5 = 0.5 → score = 0.5*2 = 1.0
        det = _build_detector(_make_fake_model(_uniform_quantiles(1, q_value=1.0)), n_dims=1)
        for _ in range(_WARMUP):
            det.update([0.0])
        score = det.update([0.0])
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_score_scales_with_residual_magnitude(self):
        # q = 0, y = v → score = v (from the identity above)
        for v in [0.5, 2.0, 10.0]:
            det = _build_detector(_make_fake_model(_uniform_quantiles(1, q_value=0.0)), n_dims=1)
            for _ in range(_WARMUP):
                det.update([0.0])
            score = det.update([v])
            self.assertAlmostEqual(score, v, places=4, msg=f"v={v}")

    def test_score_averaged_across_dimensions(self):
        # 2 dims: both q=0, obs=[1, 3] → per-dim scores = 1.0, 3.0 → mean = 2.0
        n_dims = 2
        det = _build_detector(_make_fake_model(_uniform_quantiles(n_dims, q_value=0.0)), n_dims=n_dims)
        for _ in range(_WARMUP):
            det.update([0.0] * n_dims)
        score = det.update([1.0, 3.0])
        self.assertAlmostEqual(score, 2.0, places=5)


class TestForecastCadence(unittest.TestCase):
    """Verify the model is called at the right times."""

    def test_forecast_called_once_at_warmup(self):
        fake_model = _make_fake_model(_uniform_quantiles(1))
        det = _build_detector(fake_model, n_dims=1)

        for _ in range(_WARMUP):
            det.update([0.0])

        self.assertEqual(fake_model.forecast.call_count, 1)

    def test_forecast_called_every_stride_after_warmup(self):
        fake_model = _make_fake_model(_uniform_quantiles(1))
        det = _build_detector(fake_model, n_dims=1)

        for _ in range(_WARMUP + _STRIDE * 3):
            det.update([0.0])

        # 1 at warmup + 3 reforecasts
        self.assertEqual(fake_model.forecast.call_count, 4)

    def test_pending_quantiles_consumed_in_order(self):
        # slot i: all quantiles = 0, observation = i+1 → CRPS = i+1
        # Build quantile array (n_dims=1, _STRIDE, num_q) where slot i has q=0
        # We'll observe i+1 at each slot and verify scores are 1, 2, 3, 4, 5
        fake_model = _make_fake_model(_uniform_quantiles(1, q_value=0.0))
        det = _build_detector(fake_model, n_dims=1)

        for _ in range(_WARMUP):
            det.update([0.0])

        for i, expected in enumerate(range(1, _STRIDE + 1)):
            score = det.update([float(expected)])
            self.assertAlmostEqual(score, float(expected), places=4, msg=f"slot {i}")

    def test_forecast_input_batches_dimensions(self):
        n_dims = 3
        fake_model = _make_fake_model(_uniform_quantiles(n_dims))
        det = _build_detector(fake_model, n_dims=n_dims)

        for _ in range(_WARMUP):
            det.update([0.0] * n_dims)

        context = fake_model.forecast.call_args.kwargs["context"]
        # context should be (n_dims, context_len) — one row per dimension
        self.assertEqual(context.shape[0], n_dims)


if __name__ == "__main__":
    unittest.main()
