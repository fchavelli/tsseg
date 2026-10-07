"""Change point selection of ChangeFinderDetector without K."""

import numpy as np

from tsseg.algorithms import ChangeFinderDetector
from tsseg.algorithms.utils import consensus_change_points


def _steps(n_per=400, levels=(0.0, 3.0, -1.0, 2.0), rng=0):
    rng = np.random.default_rng(rng)
    return np.concatenate([rng.normal(m, 1.0, n_per) for m in levels])


def test_threshold_factor_keeps_fewer_peaks():
    x = _steps()
    loose = ChangeFinderDetector(threshold_factor=1.0).fit_predict(x)
    strict = ChangeFinderDetector(threshold_factor=4.0).fit_predict(x)
    assert len(strict) <= len(loose)
    assert set(strict) <= set(loose)


def test_min_distance_fraction():
    x = _steps()
    pred = ChangeFinderDetector(
        threshold_factor=0.5, min_distance_fraction=0.1
    ).fit_predict(x)
    assert len(pred) > 0
    assert np.all(np.diff(pred) >= 0.1 * len(x))


def test_consensus_keeps_change_points_shared_by_channels():
    # Channel 0 gets a strong extra change of its own: nearby peaks of different
    # channels form one change point, kept only if enough channels see it.
    per_channel = [np.array([100, 500]), np.array([103]), np.array([98]), np.array([])]
    assert list(consensus_change_points(per_channel, tolerance=10, consensus=0.5)) == [
        100
    ]
    assert list(consensus_change_points(per_channel, tolerance=10, consensus=0.0)) == [
        100,
        500,
    ]
    assert list(consensus_change_points(per_channel, tolerance=10, consensus=1.0)) == []


def test_consensus_counts_channels_not_points():
    # Two peaks of the same channel are one vote.
    per_channel = [np.array([100, 105]), np.array([]), np.array([])]
    assert list(consensus_change_points(per_channel, tolerance=10, consensus=0.5)) == []


def test_ensembling_without_k_merges_the_channels():
    rng = np.random.default_rng(1)
    shared = _steps(levels=(0.0, 4.0))
    X = np.column_stack([shared + rng.normal(0, 0.2, shared.size) for _ in range(4)])
    X[200:260, 0] += 6.0  # a burst on channel 0 only
    det = ChangeFinderDetector(multivariate_strategy="ensembling", threshold_factor=3.0)
    pred = det.fit_predict(X)
    assert np.min(np.abs(pred - 400)) <= 40
    # One change point per group of coinciding detections, not one per channel.
    per_channel = [
        det._select_peaks(det._changefinder_scores(X[:, c])) for c in range(4)
    ]
    assert len(pred) < sum(len(p) for p in per_channel)
    # Requiring every channel drops the burst seen by channel 0 only.
    strict = ChangeFinderDetector(
        multivariate_strategy="ensembling", threshold_factor=3.0, consensus=1.0
    ).fit_predict(X)
    assert len(strict) <= len(pred)
    assert not np.any(np.abs(strict - 230) <= 40)


def test_guided_ensembling_returns_n_cps():
    X = np.column_stack([_steps(rng=r) for r in range(3)])
    pred = ChangeFinderDetector(
        multivariate_strategy="ensembling", n_cps=3
    ).fit_predict(X)
    assert len(pred) <= 3
