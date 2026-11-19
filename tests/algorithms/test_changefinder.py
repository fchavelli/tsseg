"""Change point selection of ChangeFinderDetector."""

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


def test_guided_completes_with_peaks_below_the_threshold():
    x = _steps()
    unsupervised = ChangeFinderDetector().fit_predict(x)
    k = len(unsupervised) + 2
    guided = ChangeFinderDetector(n_cps=k).fit_predict(x)
    assert len(guided) == k
    assert set(unsupervised) <= set(guided)
    assert np.all(np.diff(guided) >= 10)


def test_guided_output_does_not_depend_on_the_threshold():
    x = _steps()
    for k in (1, 3, 6):
        default = ChangeFinderDetector(n_cps=k).fit_predict(x)
        unreachable = ChangeFinderDetector(n_cps=k, threshold=1e9).fit_predict(x)
        assert len(default) == k
        np.testing.assert_array_equal(unreachable, default)


def _fixed_channel_peaks(monkeypatch, per_channel):
    """Make each channel of the ensembling strategy return the given peaks."""
    peaks = iter([np.asarray(p, dtype=np.int64) for p in per_channel])
    monkeypatch.setattr(
        ChangeFinderDetector, "_select_peaks", lambda self, scores: next(peaks)
    )


def test_default_tolerance_is_a_fraction_of_the_length():
    assert ChangeFinderDetector().get_params()["tolerance"] == 0.01


def test_guided_ensembling_groups_nearby_channel_peaks_by_default(monkeypatch):
    # Two channels see the same two changes a few samples apart, a third one
    # sees another change alone. Exact matching (tolerance 0) gives every peak
    # one vote and returns the two earliest ones, 100 and 104.
    _fixed_channel_peaks(monkeypatch, [[100, 600], [104, 603], [350]])
    X = np.random.default_rng(0).normal(size=(1000, 3))
    pred = ChangeFinderDetector(
        multivariate_strategy="ensembling", n_cps=2
    ).fit_predict(X)
    assert len(pred) == 2
    assert abs(pred[0] - 102) <= 2
    assert abs(pred[1] - 601) <= 2


def test_ensembling_without_k_merges_within_the_default_tolerance(monkeypatch):
    # 0.01 of 2000 samples = 20 samples, more than ``min_distance`` (10).
    _fixed_channel_peaks(monkeypatch, [[100], [118], [400]])
    X = np.random.default_rng(0).normal(size=(2000, 3))
    pred = ChangeFinderDetector(multivariate_strategy="ensembling").fit_predict(X)
    assert len(pred) == 2
    assert 100 <= pred[0] <= 118
    assert pred[1] == 400


def test_ensembling_without_k_zero_tolerance_means_min_distance(monkeypatch):
    _fixed_channel_peaks(monkeypatch, [[100], [118], [400]])
    X = np.random.default_rng(0).normal(size=(2000, 3))
    pred = ChangeFinderDetector(
        multivariate_strategy="ensembling", tolerance=0
    ).fit_predict(X)
    np.testing.assert_array_equal(pred, [100, 118, 400])
