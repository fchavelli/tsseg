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
