"""Tests for DWT-MLEAD anomaly detector.

Run with: uv run --project . pytest tests.py -v
"""

import math

import numpy as np

from detector import DwtMlead


def test_name():
    assert DwtMlead.name() == "DWT-MLEAD"


def test_warmup_returns_nan():
    det = DwtMlead(n_dimensions=1)
    assert math.isnan(det.update([1.0]))


def test_returns_float_after_warmup():
    det = DwtMlead(n_dimensions=1)
    scores = [det.update([0.0]) for _ in range(200)]
    non_nan = [s for s in scores if not math.isnan(s)]
    assert len(non_nan) > 0
    assert all(isinstance(s, float) for s in non_nan)


def test_constant_signal_no_anomaly():
    """Constant input → zero detail at every level → no threshold crossings."""
    det = DwtMlead(n_dimensions=1)
    scores = [det.update([5.0]) for _ in range(300)]
    post_warmup = [s for s in scores if not math.isnan(s)]
    assert len(post_warmup) > 0
    assert all(s == 0.0 for s in post_warmup)


def test_spike_detected():
    """A large spike in a flat signal exceeds the threshold."""
    det = DwtMlead(n_dimensions=1)
    for _ in range(200):
        det.update([0.0])
    score = det.update([100.0])
    assert not math.isnan(score)
    assert score > 0.0


def test_larger_spike_scores_higher():
    """Anomaly score is monotone in spike magnitude (trained on noise so variance is non-zero)."""
    rng = np.random.default_rng(0)
    baseline = [float(x) for x in rng.normal(0, 1, 500)]

    def spike_score(magnitude: float) -> float:
        det = DwtMlead(n_dimensions=1)
        for x in baseline:
            det.update([x])
        return det.update([magnitude])

    assert spike_score(5.0) < spike_score(20.0) < spike_score(100.0)


def test_false_positive_rate_low():
    """Gaussian noise should trigger anomalies in fewer than 10% of post-warmup samples."""
    rng = np.random.default_rng(42)
    det = DwtMlead(n_dimensions=1)
    scores = [det.update([float(x)]) for x in rng.normal(0, 1, 1000)]
    valid = [s for s in scores if not math.isnan(s)]
    flagged = sum(1 for s in valid if s > 0.0)
    assert len(valid) > 0
    assert flagged / len(valid) < 0.10


def test_score_zero_for_normal_inlier():
    """A point close to the learned mean scores 0 (below threshold)."""
    rng = np.random.default_rng(7)
    det = DwtMlead(n_dimensions=1)
    for x in rng.normal(0, 1, 500):
        det.update([float(x)])
    # Feed the mean itself — detail will be tiny, well within threshold
    score = det.update([0.0])
    assert score == 0.0
