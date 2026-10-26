"""Tests specific to ``GreedyGaussianDetector``."""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import GreedyGaussianDetector
from tsseg.algorithms.ggs.detector import _GGS


def _three_segments() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate(
        [
            rng.normal(0.0, 1.0, (150, 2)),
            rng.normal(3.0, 1.0, (100, 2)),
            rng.normal(-2.0, 0.5, (150, 2)),
        ]
    )


@pytest.mark.parametrize("univariate", [False, True])
def test_returns_change_point_indices(univariate):
    """The output is the change point indices, not one label per time point.

    The detector declares ``returns_dense=True``: it used to return one
    segment label per time point.
    """
    X = _three_segments()
    if univariate:
        X = X[:, 0]
    change_points = GreedyGaussianDetector(k_max=2, random_state=0).fit_predict(X)
    assert isinstance(change_points, np.ndarray)
    assert np.issubdtype(change_points.dtype, np.integer)
    np.testing.assert_array_equal(change_points, [150, 250])


def test_k_max_bounds_the_number_of_change_points():
    X = _three_segments()
    change_points = GreedyGaussianDetector(k_max=1, random_state=0).fit_predict(X)
    assert change_points.tolist() == [250]


def _shifted_mean_segment(offset: float = 0.0) -> np.ndarray:
    """Three channels, mean change at 50, away from zero (left mean matters)."""
    rng = np.random.default_rng(1)
    return offset + np.concatenate(
        [rng.normal(5.0, 1.0, (50, 3)), rng.normal(6.5, 1.0, (70, 3))]
    )


@pytest.mark.parametrize("lamb", [0.1, 1.0])
def test_split_objective_matches_direct_log_likelihood(lamb):
    """The incremental split search scores each split as the direct objective.

    ``add_new_change_point`` updates the left and right means and covariances
    in one pass. Its best split and gain must match the brute-force search
    over ``log_likelihood(left) + log_likelihood(right)``. The left mean used
    to start from ``data[0] / n_channels`` instead of ``data[0]``, which only
    agreed in the univariate case.
    """
    data = _shifted_mean_segment()
    ggs = _GGS(lamb=lamb)
    index, gain = ggs.add_new_change_point(data)
    m = len(data)
    direct = {
        i: ggs.log_likelihood(data[:i]) + ggs.log_likelihood(data[i:])
        for i in range(2, m - 1)
    }
    best = min(direct, key=direct.get)
    assert index == best
    np.testing.assert_allclose(gain, direct[best] - ggs.log_likelihood(data), rtol=1e-9)


@pytest.mark.parametrize("univariate", [False, True])
def test_change_points_do_not_depend_on_a_constant_offset(univariate):
    """The Gaussian likelihood only sees covariances: shifting X changes nothing."""
    X = _three_segments()
    if univariate:
        X = X[:, 0]
    expected = GreedyGaussianDetector(k_max=2, random_state=0).fit_predict(X)
    for offset in (10.0, 100.0):
        shifted = GreedyGaussianDetector(k_max=2, random_state=0).fit_predict(
            X + offset
        )
        np.testing.assert_array_equal(shifted, expected)


@pytest.mark.parametrize("data", ["three_channels", "two_channels"])
def test_zero_lamb_is_refused_on_a_multivariate_series(data):
    """``lamb=0`` leaves the covariance of a two-point segment singular.

    The split search scores a left part of two points first, whose empirical
    covariance has rank one at most: with more than one channel and no
    regularisation, the objective is unbounded and ``np.linalg.cholesky``
    used to raise ``LinAlgError: Matrix is not positive definite`` while
    predicting. ``fit`` now refuses it with a message that names ``lamb``.
    """
    X = _shifted_mean_segment() if data == "three_channels" else _three_segments()
    with pytest.raises(ValueError, match="lamb"):
        GreedyGaussianDetector(lamb=0.0).fit(X)


def test_zero_lamb_on_a_univariate_series_with_equal_values_names_lamb():
    """A segment of equal values has a zero variance, singular without ``lamb``."""
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.integers(0, 3, 100), rng.integers(5, 8, 100)])
    detector = GreedyGaussianDetector(k_max=3, lamb=0.0, max_shuffles=5).fit(X)
    with pytest.raises(np.linalg.LinAlgError, match="lamb"):
        detector.predict(X)


def test_zero_lamb_still_runs_on_a_continuous_univariate_series():
    """Without repeated values, a univariate series keeps the unregularised fit."""
    X = _three_segments()[:, 0]
    change_points = GreedyGaussianDetector(
        k_max=2, lamb=0.0, random_state=0
    ).fit_predict(X)
    np.testing.assert_array_equal(change_points, [150, 250])
