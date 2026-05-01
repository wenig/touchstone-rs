"""Tests for TinyTimeMixer scoring logic.

Scoring summary
---------------
- Buffer fills as a deque (maxlen=_CONTEXT_LENGTH=1024).
- No score is produced until len(buffer) >= _WARMUP (512); those updates return NaN.
- After point 511 is appended (buffer length hits _WARMUP), a forecast of shape
  (_STRIDE=5, n_dims) is computed and stored as pending_predictions.
- From point 512 onward, each update consumes one pending prediction:
    score = mean(abs(observation - pending_predictions[pending_idx]))
  and increments pending_idx.
- When pending_idx reaches _STRIDE (5), a fresh forecast is triggered and pending_idx
  resets to 0. So forecasts run every _STRIDE points after warmup.
"""

import math
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Stub out heavy dependencies so we can import detector without a GPU / HF hub
# ---------------------------------------------------------------------------

# touchstone_py: Detector is just a base class the detector inherits from
_touchstone_stub = types.ModuleType("touchstone_py")
_touchstone_stub.Detector = object
_touchstone_stub.run_cli = lambda cls: None
sys.modules.setdefault("touchstone_py", _touchstone_stub)

# model: needs TinyTimeMixerForPrediction defined before detector imports it
_model_stub = types.ModuleType("model")
_model_stub.TinyTimeMixerForPrediction = MagicMock()
sys.modules.setdefault("model", _model_stub)

# Now it is safe to import the detector module
import importlib
import detector as _detector_module

_WARMUP = _detector_module._WARMUP
_STRIDE = _detector_module._STRIDE
_CONTEXT_LENGTH = _detector_module._CONTEXT_LENGTH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_model(predictions: np.ndarray):
    """Return a mock whose __call__ returns a result with .prediction_outputs.

    predictions: shape (prediction_length, n_dims) — what the model always returns.
    """
    output = MagicMock()
    # model returns (1, prediction_length, n_dims); detector slices [0, :_STRIDE]
    output.prediction_outputs = torch.tensor(predictions[np.newaxis], dtype=torch.float32)

    fake_model = MagicMock()
    fake_model.return_value = output
    fake_model.to.return_value = fake_model
    fake_model.eval.return_value = fake_model
    return fake_model


def _build_detector(fake_model, n_dims=1):
    """Instantiate TinyTimeMixer with a pre-built fake model."""
    _model_stub.TinyTimeMixerForPrediction = MagicMock(
        from_pretrained=MagicMock(return_value=fake_model)
    )
    # Reload so the module picks up the patched stub
    importlib.reload(_detector_module)
    return _detector_module.TinyTimeMixer(n_dimensions=n_dims)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWarmup(unittest.TestCase):
    def setUp(self):
        preds = np.zeros((_STRIDE * 2, 1), dtype=np.float32)
        self.det = _build_detector(_make_fake_model(preds), n_dims=1)

    def test_all_nan_during_warmup(self):
        for i in range(_WARMUP):
            score = self.det.update([float(i)])
            self.assertTrue(math.isnan(score), f"Expected NaN at step {i}, got {score}")

    def test_first_post_warmup_score_is_not_nan(self):
        for i in range(_WARMUP):
            self.det.update([0.0])
        score = self.det.update([0.0])
        self.assertFalse(math.isnan(score))


class TestScoreCalculation(unittest.TestCase):
    """Verify MAE is computed correctly against known predictions."""

    def _run_to_first_score(self, n_dims, pred_value, obs_value):
        """Feed _WARMUP zeros, then one observation, return the first real score."""
        # predictions: all pred_value
        preds = np.full((_STRIDE * 2, n_dims), pred_value, dtype=np.float32)
        det = _build_detector(_make_fake_model(preds), n_dims=n_dims)

        for _ in range(_WARMUP):
            det.update([0.0] * n_dims)

        return det.update([obs_value] * n_dims)

    def test_zero_error_when_prediction_matches_observation(self):
        score = self._run_to_first_score(n_dims=1, pred_value=3.0, obs_value=3.0)
        self.assertAlmostEqual(score, 0.0)

    def test_scalar_mae_single_dimension(self):
        # MAE = |obs - pred| = |5 - 2| = 3
        score = self._run_to_first_score(n_dims=1, pred_value=2.0, obs_value=5.0)
        self.assertAlmostEqual(score, 3.0, places=5)

    def test_mean_mae_multi_dimension(self):
        # 3 dims: obs=[1,2,3], pred=[0,0,0] → MAE = mean(1,2,3) = 2.0
        n_dims = 3
        preds = np.zeros((_STRIDE * 2, n_dims), dtype=np.float32)
        det = _build_detector(_make_fake_model(preds), n_dims=n_dims)

        for _ in range(_WARMUP):
            det.update([0.0] * n_dims)

        score = det.update([1.0, 2.0, 3.0])
        self.assertAlmostEqual(score, 2.0, places=5)

    def test_negative_residual_uses_absolute_value(self):
        # obs < pred: |2 - 5| = 3, not -3
        score = self._run_to_first_score(n_dims=1, pred_value=5.0, obs_value=2.0)
        self.assertAlmostEqual(score, 3.0, places=5)


class TestForecastCadence(unittest.TestCase):
    """Verify the model is called at the right times."""

    def test_forecast_called_once_at_warmup(self):
        preds = np.zeros((_STRIDE * 2, 1), dtype=np.float32)
        fake_model = _make_fake_model(preds)
        det = _build_detector(fake_model, n_dims=1)

        for i in range(_WARMUP):
            det.update([0.0])

        self.assertEqual(fake_model.call_count, 1)

    def test_forecast_called_every_stride_after_warmup(self):
        preds = np.zeros((_STRIDE * 2, 1), dtype=np.float32)
        fake_model = _make_fake_model(preds)
        det = _build_detector(fake_model, n_dims=1)

        for _ in range(_WARMUP + _STRIDE * 3):
            det.update([0.0])

        # 1 at warmup + 3 reforecasts
        self.assertEqual(fake_model.call_count, 4)

    def test_pending_predictions_consumed_in_order(self):
        # Each of the _STRIDE slots has a distinct value so we can verify order
        slot_values = np.arange(1, _STRIDE + 1, dtype=np.float32).reshape(-1, 1)
        # prediction_length must be >= _STRIDE; pad with zeros after
        preds = np.vstack([slot_values, np.zeros((_STRIDE, 1), dtype=np.float32)])
        det = _build_detector(_make_fake_model(preds), n_dims=1)

        for _ in range(_WARMUP):
            det.update([0.0])

        # observation=0 each time → score = |0 - pred_slot| = slot value
        for i, expected in enumerate(range(1, _STRIDE + 1)):
            score = det.update([0.0])
            self.assertAlmostEqual(score, float(expected), places=5, msg=f"slot {i}")


if __name__ == "__main__":
    unittest.main()
