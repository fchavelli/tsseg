"""Tests for ChangeForestDetector (wrapper of the changeforest package)."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

changeforest = pytest.importorskip("changeforest")

from tsseg.algorithms import ChangeForestDetector  # noqa: E402


def _cic_example():
    """Change in covariance (CIC) series of the package README (n=600, d=5)."""
    rng = np.random.default_rng(12)
    sigma = np.full((5, 5), 0.7)
    np.fill_diagonal(sigma, 1)
    return np.concatenate(
        (
            rng.normal(0, 1, (200, 5)),
            rng.multivariate_normal(np.zeros(5), sigma, 200, method="cholesky"),
            rng.normal(0, 1, (200, 5)),
        ),
        axis=0,
    )


def test_readme_example():
    """The package README finds the covariance changes at 200 and 400."""
    cps = ChangeForestDetector().fit_predict(_cic_example())
    np.testing.assert_array_equal(cps, [200, 400])
    assert cps.dtype == np.int64


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"segmentation": "sbs"},
        {"segmentation": "wbs", "random_state": 3},
        {"method": "knn"},
        {"method": "change_in_mean"},
        {"method": "change_in_mean", "minimal_gain_to_split": 5.0},
        {"random_forest_max_features": None, "random_forest_max_depth": None},
        {"random_forest_n_estimators": 20, "model_selection_alpha": 0.1},
        {"minimal_relative_segment_length": 0.05},
    ],
)
def test_unsupervised_matches_package(params):
    """Without n_cps, the detector returns the package's split points."""
    X = _cic_example()
    detector = ChangeForestDetector(**params)
    control = detector._control(detector.minimal_relative_segment_length)
    expected = changeforest.changeforest(
        X, detector.method, detector.segmentation, control
    ).split_points()
    np.testing.assert_array_equal(detector.fit_predict(X), sorted(expected))


def test_guided_node_scores_match_package_tree():
    """A segment scored alone gets the split and gain of the package's tree node."""
    X = _cic_example()
    tree = changeforest.changeforest(X, "random_forest", "bs")
    detector = ChangeForestDetector()
    min_length = math.ceil(0.01 * len(X))
    for node in (tree, tree.left, tree.right, tree.left.left, tree.left.right):
        split, gain = detector._best_split(X, node.start, node.stop, min_length)
        assert split == node.best_split
        assert gain == pytest.approx(node.max_gain, rel=1e-12)


@pytest.mark.parametrize("segmentation", ["bs", "sbs", "wbs"])
def test_guided_returns_requested_number(segmentation):
    X = _cic_example()
    previous: set[int] = set()
    for k in (1, 2, 4):
        cps = ChangeForestDetector(n_cps=k, segmentation=segmentation).fit_predict(X)
        assert len(cps) == k
        assert np.all(np.diff(cps) > 0)
        assert cps[0] >= 6 and cps[-1] <= len(X) - 6
        # Best-first splitting: the K change points extend the K - 1 ones.
        assert previous <= set(cps.tolist())
        previous = set(cps.tolist())


def test_guided_two_change_points():
    cps = ChangeForestDetector(n_cps=2).fit_predict(_cic_example())
    np.testing.assert_array_equal(cps, [200, 400])


def test_guided_zero_change_points():
    cps = ChangeForestDetector(n_cps=0).fit_predict(_cic_example())
    assert cps.shape == (0,)
    assert cps.dtype == np.int64


def test_guided_warns_when_segments_too_short():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    detector = ChangeForestDetector(n_cps=30, minimal_relative_segment_length=0.1)
    with pytest.warns(UserWarning, match="found only"):
        cps = detector.fit_predict(X)
    # Minimal segment length ceil(0.1 * 40) = 4: at most 9 change points.
    assert 0 < len(cps) <= 9
    assert np.all(np.diff(np.concatenate(([0], cps, [40]))) >= 4)


def test_univariate_mean_shift():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(0, 1, 300), rng.normal(3, 1, 300)])
    cps = ChangeForestDetector().fit_predict(x)
    assert len(cps) == 1
    assert abs(int(cps[0]) - 300) <= 5


def test_series_shorter_than_two_segments():
    # The package splits a segment only when it has more than 2 * ceil(delta * n)
    # points; below that, wild binary segmentation would loop forever.
    X = np.arange(12, dtype=float).reshape(6, 2)
    for segmentation in ("bs", "sbs", "wbs"):
        detector = ChangeForestDetector(
            minimal_relative_segment_length=0.45, segmentation=segmentation
        )
        assert detector.fit_predict(X).shape == (0,)
    assert ChangeForestDetector(n_cps=1).fit_predict(X[:1]).shape == (0,)


@pytest.mark.parametrize(
    "params",
    [
        {"minimal_relative_segment_length": 0.5},
        {"minimal_relative_segment_length": 0.0},
        {"model_selection_alpha": 1.0},
        {"seeded_segments_alpha": 1.0},
        {"method": "svm"},
        {"segmentation": "pelt"},
        {"random_forest_max_features": "log2"},
        {"n_cps": -1},
    ],
)
def test_invalid_parameters_rejected_before_the_package(params):
    """Invalid values raise ValueError instead of a Rust panic."""
    with pytest.raises(ValueError):
        ChangeForestDetector(**params).fit_predict(_cic_example())


def test_missing_package_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "changeforest", None)
    with pytest.raises(ImportError, match=r"tsseg\[changeforest\]"):
        ChangeForestDetector().fit_predict(_cic_example())
