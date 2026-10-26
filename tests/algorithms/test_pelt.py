"""Tests specific to :class:`~tsseg.algorithms.PeltDetector`."""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import PeltDetector


@pytest.fixture
def X():
    rng = np.random.default_rng(0)
    return np.r_[rng.normal(0, 1, 150), rng.normal(5, 1, 150)]


def test_get_fitted_params_after_fit_skips_change_points(X):
    params = PeltDetector().fit(X).get_fitted_params()
    assert "change_points_" not in params
    assert "metadata_" in params


def test_get_fitted_params_after_predict_lists_change_points(X):
    detector = PeltDetector().fit(X)
    change_points = detector.predict(X)
    params = detector.get_fitted_params()
    np.testing.assert_array_equal(params["change_points_"], change_points)


def test_change_points_before_predict_is_not_set_yet(X):
    detector = PeltDetector().fit(X)
    assert not hasattr(detector, "change_points_")
    # Still a RuntimeError, as before, for callers that catch it.
    with pytest.raises(RuntimeError, match="Predict must be called"):
        _ = detector.change_points_
