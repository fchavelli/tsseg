"""Tests specific to ``GreedyGaussianDetector``."""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import GreedyGaussianDetector


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
