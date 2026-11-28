"""TIRE postprocessing uses time units independently of window subsampling."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsseg.algorithms import TireDetector  # noqa: E402
from tsseg.algorithms.tire import utils  # noqa: E402


@pytest.fixture
def features(monkeypatch):
    td, fd = np.random.default_rng(12).normal(size=(2, 100, 2))
    fd *= 5
    monkeypatch.setattr(
        TireDetector, "_compute_latent_features", lambda self, X: (td, fd)
    )
    return td, fd


def _original_feature_scores(features, domain, w):
    """Frozen pre-migration postprocessing, in feature-row coordinates."""
    td, fd = features
    if domain == "both":
        beta = np.quantile(utils.distance(td, w), 0.95)
        alpha = np.quantile(utils.distance(fd, w), 0.95)
        original_features = np.concatenate((td * alpha, fd * beta), axis=1)
    else:
        original_features = td if domain == "TD" else fd
    smoothed = utils.matched_filter(original_features, w)
    distances = utils.matched_filter(utils.distance(smoothed, w), w)
    scores = np.asarray(utils.new_peak_prominences(distances)[0])
    scores /= np.max(scores) + 1e-8
    return scores


@pytest.mark.parametrize("domain", ["TD", "FD", "both"])
def test_unit_stride_preserves_the_original_scores_exactly(features, domain):
    """Freeze the pre-migration pipeline as the unit-stride reference."""
    w, n = 16, 115
    scores = _original_feature_scores(features, domain, w)
    expected = np.interp(
        np.arange(n), np.arange(scores.size) + w, scores, left=0.0, right=0.0
    )
    actual = TireDetector(window_size=w, domain=domain)._run_pipeline(np.zeros((n, 1)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("domain", ["TD", "FD", "both"])
@pytest.mark.parametrize("w,stride", [(16, 2), (16, 4), (20, 3), (16, 16)])
def test_same_feature_geometry_gives_same_scores_in_time_coordinates(
    features, domain, w, stride
):
    """Compare with a unit-stride detector in feature coordinates.

    Window pairs start at least one physical window apart. Their boundary is
    midway between the end of the earlier window and start of the later one.
    The reference covers both domain weights and both smoothing operations.
    """
    starts = np.arange(len(features[0])) * stride
    lag = np.flatnonzero(starts >= w)[0]
    scores = _original_feature_scores(features, domain, lag)
    midpoints = ((starts[:-lag] + w) + starts[lag:]) / 2
    n = starts[-1] + w
    expected = np.interp(np.arange(n), midpoints, scores, left=0.0, right=0.0)
    actual = TireDetector(window_size=w, stride=stride, domain=domain)._run_pipeline(
        np.zeros((n, 1))
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("w,stride", [(16, 1), (16, 2), (16, 4), (20, 3), (16, 16)])
def test_window_mean_step_scores_remain_centered_on_the_boundary(
    monkeypatch, w, stride
):
    def window_means(self, X):
        windows = self._prepare_windows(X)
        return windows.mean(axis=1, keepdims=True), None

    monkeypatch.setattr(TireDetector, "_compute_latent_features", window_means)
    signal = np.repeat([0.0, 1.0], 300)
    detector = TireDetector(window_size=w, stride=stride, domain="TD", n_segments=2)
    cps = detector.fit_predict(signal)
    assert detector.change_scores_.shape == signal.shape
    assert np.isfinite(detector.change_scores_).all()
    assert len(cps) == 1
    assert abs(cps[0] - 300) <= stride / 2


@pytest.mark.parametrize("domain", ["TD", "FD", "both"])
@pytest.mark.parametrize("stride", [1, 3, 16])
def test_no_comparable_window_pair_has_zero_scores(monkeypatch, domain, stride):
    features = np.ones((1, 2))
    monkeypatch.setattr(
        TireDetector, "_compute_latent_features", lambda self, X: (features, features)
    )
    detector = TireDetector(window_size=16, stride=stride, domain=domain)
    np.testing.assert_array_equal(
        detector._run_pipeline(np.zeros((17, 1))), np.zeros(17)
    )
