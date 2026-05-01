"""Tests for Toto scoring logic.

Scoring summary
---------------
- Buffer fills as a deque (maxlen=_CONTEXT_LENGTH=1024).
- No score until len(buffer) >= _WARMUP (512); those updates return NaN.
- After warmup, each update consumes one column of pending_quantiles:
    pending_quantiles shape: (9, n_dims, _STRIDE)
    score = CRPS(observation, pending_quantiles[:, :, pending_idx])
- CRPS is the pinball-loss approximation:
    diff = y - q  per (quantile_level, dim)
    loss = tau*diff if diff >= 0 else (tau-1)*diff
    crps = 2 * mean(loss)
- A new forecast is triggered every _STRIDE points after warmup.
"""

import math
import sys
import types
import unittest
from unittest.mock import MagicMock
import importlib

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Stubs so detector.py can be imported without torch/HF/toto2
# ---------------------------------------------------------------------------

_touchstone_stub = types.ModuleType("touchstone_py")
_touchstone_stub.Detector = object
_touchstone_stub.run_cli = lambda cls: None
sys.modules.setdefault("touchstone_py", _touchstone_stub)

_toto2_stub = types.ModuleType("toto2")
_toto2_stub.Toto2Model = MagicMock()
sys.modules.setdefault("toto2", _toto2_stub)

import detector as _det_mod

_WARMUP = _det_mod._WARMUP
_STRIDE = _det_mod._STRIDE
_CONTEXT_LENGTH = _det_mod._CONTEXT_LENGTH
_QUANTILE_LEVELS = _det_mod._QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(quantile_values: np.ndarray):
    """Return a mock whose .forecast() returns a known quantile tensor.

    quantile_values: shape (9, n_dims, horizon) — what the model always returns.
    """
    # model returns (9, batch=1, n_dims, horizon)
    raw = quantile_values[:, np.newaxis, :, :]
    tensor = torch.tensor(raw, dtype=torch.float32)

    fake_model = MagicMock()
    fake_model.forecast.return_value = tensor
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    return fake_model


def _build_detector(fake_model, n_dims=1):
    _toto2_stub.Toto2Model = MagicMock(from_pretrained=MagicMock(return_value=fake_model))
    importlib.reload(_det_mod)
    return _det_mod.Toto(n_dimensions=n_dims)


def _zero_quantiles(n_dims, horizon=None):
    if horizon is None:
        horizon = _STRIDE * 2
    return np.zeros((9, n_dims, horizon), dtype=np.float32)


# ---------------------------------------------------------------------------
# Warmup tests
# ---------------------------------------------------------------------------

class TestWarmup(unittest.TestCase):
    def setUp(self):
        self.det = _build_detector(_make_fake_model(_zero_quantiles(1)), n_dims=1)

    def test_all_nan_during_warmup(self):
        for i in range(_WARMUP):
            score = self.det.update([0.0])
            self.assertTrue(math.isnan(score), f"Expected NaN at step {i}, got {score}")

    def test_first_post_warmup_score_is_not_nan(self):
        for _ in range(_WARMUP):
            self.det.update([0.0])
        score = self.det.update([0.0])
        self.assertFalse(math.isnan(score))


# ---------------------------------------------------------------------------
# CRPS correctness
# ---------------------------------------------------------------------------

class TestCRPS(unittest.TestCase):
    """Unit-test _crps directly against closed-form results."""

    def _crps(self, y, quantiles):
        return _det_mod.Toto._crps(
            np.array(y, dtype=np.float32),
            np.array(quantiles, dtype=np.float32),
        )

    def test_zero_error_when_prediction_matches(self):
        # y == all quantiles → every diff is 0 → CRPS = 0
        q = np.zeros((9, 1), dtype=np.float32)
        self.assertAlmostEqual(self._crps([0.0], q), 0.0)

    def test_known_value_y_above_all_quantiles(self):
        # y=1, all q=0 → diff=1 (positive) for all tau
        # loss_i = tau_i * 1; mean(tau) = 0.5; CRPS = 2 * 0.5 = 1.0
        q = np.zeros((9, 1), dtype=np.float32)
        self.assertAlmostEqual(self._crps([1.0], q), 1.0, places=5)

    def test_known_value_y_below_all_quantiles(self):
        # y=0, all q=1 → diff=-1 (negative) for all tau
        # loss_i = (tau_i - 1) * (-1) = (1 - tau_i); mean(1-tau) = 0.5; CRPS = 1.0
        q = np.ones((9, 1), dtype=np.float32)
        self.assertAlmostEqual(self._crps([0.0], q), 1.0, places=5)

    def test_crps_is_symmetric_around_quantile_set(self):
        # Overshooting and undershooting by the same amount should give equal CRPS
        q = np.zeros((9, 1), dtype=np.float32)
        above = self._crps([1.0], q)
        q_high = np.ones((9, 1), dtype=np.float32)
        below = self._crps([0.0], q_high)
        self.assertAlmostEqual(above, below, places=5)

    def test_multi_dim_means_over_all_dimensions(self):
        # n_dims=2; y=[1,0]; all q=0
        # dim0: diff=1, all losses = tau_i * 1
        # dim1: diff=0, all losses = 0
        # mean across (9,2) = mean(tau)/2 = 0.5/2 = 0.25; CRPS = 0.5
        q = np.zeros((9, 2), dtype=np.float32)
        score = self._crps([1.0, 0.0], q)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_all_dims_contribute_to_score(self):
        # Score from dim1-only error should equal score from dim0-only error (by symmetry)
        q = np.zeros((9, 2), dtype=np.float32)
        score_dim0 = self._crps([1.0, 0.0], q)
        score_dim1 = self._crps([0.0, 1.0], q)
        self.assertAlmostEqual(score_dim0, score_dim1, places=5)
        self.assertGreater(score_dim0, 0.0)

    def test_last_dim_error_is_not_ignored(self):
        # 3 dims; only the last one is wrong → score must be > 0
        q = np.zeros((9, 3), dtype=np.float32)
        score = self._crps([0.0, 0.0, 1.0], q)
        self.assertGreater(score, 0.0)


# ---------------------------------------------------------------------------
# Forecast output shape / variate coverage
# ---------------------------------------------------------------------------

class TestForecastShape(unittest.TestCase):
    """Verify that pending_quantiles covers all variates and stride steps."""

    def test_pending_quantiles_shape(self):
        n_dims = 4
        # Use exactly _STRIDE as horizon to match what the real model returns
        det = _build_detector(_make_fake_model(_zero_quantiles(n_dims, horizon=_STRIDE)), n_dims=n_dims)
        for _ in range(_WARMUP):
            det.update([0.0] * n_dims)
        self.assertIsNotNone(det._pending_quantiles)
        self.assertEqual(det._pending_quantiles.shape, (9, n_dims, _STRIDE))

    def test_each_stride_slot_consumed_in_order(self):
        # Give each stride slot a distinct quantile value so we can verify order.
        n_dims = 1
        # quantiles[q, 0, t] = t+1 for all q
        slot_vals = np.arange(1, _STRIDE + 1, dtype=np.float32)
        q = np.tile(slot_vals, (9, n_dims, 1))  # (9, 1, _STRIDE)
        # pad with zeros for a second batch of forecasts
        q_padded = np.concatenate([q, np.zeros((9, n_dims, _STRIDE), dtype=np.float32)], axis=2)
        det = _build_detector(_make_fake_model(q_padded), n_dims=n_dims)

        for _ in range(_WARMUP):
            det.update([0.0])

        # y=0, quantiles=slot_val → CRPS proportional to slot_val
        scores = [det.update([0.0]) for _ in range(_STRIDE)]
        # Scores should be strictly increasing (slot 1 < slot 2 < ...)
        for i in range(len(scores) - 1):
            self.assertLess(scores[i], scores[i + 1], f"slot {i} score not < slot {i+1}")


# ---------------------------------------------------------------------------
# Forecast cadence
# ---------------------------------------------------------------------------

class TestForecastCadence(unittest.TestCase):
    def test_forecast_called_once_at_warmup(self):
        fake = _make_fake_model(_zero_quantiles(1))
        det = _build_detector(fake, n_dims=1)
        for _ in range(_WARMUP):
            det.update([0.0])
        self.assertEqual(fake.forecast.call_count, 1)

    def test_forecast_called_every_stride_after_warmup(self):
        fake = _make_fake_model(_zero_quantiles(1))
        det = _build_detector(fake, n_dims=1)
        for _ in range(_WARMUP + _STRIDE * 3):
            det.update([0.0])
        # 1 at warmup + 3 reforecasts
        self.assertEqual(fake.forecast.call_count, 4)


# ---------------------------------------------------------------------------
# Context length padding (regression for EinopsError when ctx % patch_size != 0)
# ---------------------------------------------------------------------------

class _PatchSizeEnforcingModel:
    """Fake model that raises if context length is not divisible by patch_size."""

    def __init__(self, patch_size: int, quantile_values: np.ndarray):
        self.patch_size = patch_size
        self._quantile_values = quantile_values  # (9, n_dims, _STRIDE)
        self.call_args_list: list[dict] = []

    def to(self, *a, **kw):
        return self

    def eval(self):
        return self

    def forecast(self, inputs, horizon, **_):
        ctx = inputs["target"].shape[-1]
        self.call_args_list.append({
            "ctx": ctx,
            "mask": inputs["target_mask"].clone(),
        })
        if ctx % self.patch_size != 0:
            raise RuntimeError(
                f"EinopsError: can't divide axis of length {ctx} in chunks of {self.patch_size}"
            )
        raw = self._quantile_values[:, np.newaxis, :, :]
        return torch.tensor(raw, dtype=torch.float32)


class TestContextLengthPadding(unittest.TestCase):
    """Regression tests for the EinopsError that occurred when buffer size was not
    divisible by the model's patch size during reforecasts after warmup."""

    PATCH_SIZE = 32

    def _build_patchy_detector(self, n_dims=1):
        q = _zero_quantiles(n_dims, horizon=_STRIDE)
        fake = _PatchSizeEnforcingModel(self.PATCH_SIZE, q)
        _toto2_stub.Toto2Model = MagicMock(from_pretrained=MagicMock(return_value=fake))
        importlib.reload(_det_mod)
        return _det_mod.Toto(n_dimensions=n_dims), fake

    def test_forecast_always_called_with_context_length(self):
        # Every forecast call must pass exactly _CONTEXT_LENGTH context so the
        # model's patch-size requirement is always satisfied.
        det, fake = self._build_patchy_detector()
        for _ in range(_WARMUP + _STRIDE * 3):
            det.update([0.0])
        for call in fake.call_args_list:
            self.assertEqual(call["ctx"], _CONTEXT_LENGTH, f"model called with ctx={call['ctx']}, expected {_CONTEXT_LENGTH}")

    def test_scores_continue_past_first_reforecast(self):
        # Before the fix, the reforecast at buffer-size=_WARMUP+_STRIDE (which is
        # not divisible by 32) raised EinopsError, freezing the buffer and making
        # every subsequent update() return NaN.
        det, _ = self._build_patchy_detector()
        for _ in range(_WARMUP):
            det.update([0.0])
        # Run two full stride cycles; all scores must be non-NaN.
        scores = [det.update([0.0]) for _ in range(_STRIDE * 2)]
        for i, s in enumerate(scores):
            self.assertFalse(math.isnan(s), f"NaN score at post-warmup step {i}")

    def test_mask_marks_padding_as_unobserved(self):
        # When the buffer has fewer than _CONTEXT_LENGTH points, the model should
        # receive a mask that is False for the zero-padded prefix and True for real data.
        det, fake = self._build_patchy_detector()
        for i in range(_WARMUP):
            det.update([float(i)])
        # The warmup forecast fires when buffer == _WARMUP < _CONTEXT_LENGTH.
        self.assertEqual(len(fake.call_args_list), 1)
        mask = fake.call_args_list[0]["mask"]  # (1, n_dims, _CONTEXT_LENGTH)
        pad_len = _CONTEXT_LENGTH - _WARMUP
        # Padded prefix must be all False.
        self.assertTrue((mask[:, :, :pad_len] == False).all().item())
        # Real-data suffix must be all True.
        self.assertTrue((mask[:, :, pad_len:] == True).all().item())


if __name__ == "__main__":
    unittest.main()
