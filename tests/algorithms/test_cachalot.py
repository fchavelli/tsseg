"""Tests for Cachalot."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import lars_path

from tsseg.algorithms import CachalotDetector
from tsseg.algorithms.cachalot._core import (
    lasso_candidate_points,
    reduced_dynamic_programming,
    select_change_count,
)


def _dense_lars_candidates(y: np.ndarray, max_candidates: int) -> np.ndarray:
    n_samples = y.size
    split_points = np.arange(1, n_samples)
    design = (np.arange(n_samples)[:, None] >= split_points).astype(float)
    design -= design.mean(axis=0)
    _, active, _ = lars_path(
        design,
        y - y.mean(),
        method="lasso",
        max_iter=max_candidates,
    )
    return np.asarray(active, dtype=np.int64) + 1


def test_structured_lars_matches_dense_reference():
    rng = np.random.default_rng(7301)
    y = rng.normal(size=32)

    actual = lasso_candidate_points(y, max_candidates=10)

    assert_array_equal(actual, _dense_lars_candidates(y, max_candidates=10))


def test_reduced_dynamic_programming_matches_exhaustive_search():
    y = np.array([0.2, 0.1, 0.0, 2.0, 2.1, 1.9, -0.5, -0.4])
    candidates = np.array([2, 4, 6])
    objective_path, change_points = reduced_dynamic_programming(y, candidates)

    for n_change_points, selected in enumerate(change_points):
        expected = np.inf
        for subset in combinations(candidates, n_change_points):
            boundaries = (0, *subset, y.size)
            cost = 0.0
            for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                segment = y[start:end]
                cost += float(np.sum((segment - segment.mean()) ** 2))
            expected = min(expected, cost)
        assert_allclose(objective_path[n_change_points], expected)
        assert len(selected) == n_change_points
        selected_boundaries = (0, *selected, y.size)
        reconstructed_cost = sum(
            float(np.sum((y[start:end] - y[start:end].mean()) ** 2))
            for start, end in zip(
                selected_boundaries[:-1], selected_boundaries[1:], strict=True
            )
        )
        assert_allclose(reconstructed_cost, expected)


def test_selection_rule_matches_published_toy_objectives():
    published_objectives = np.array(
        [696.28, 249.24, 209.94, 146.29, 120.21, 118.22, 116.97, 116.66, 116.65, 116.64]
    )
    published_ratios = np.array(
        [0.3580, 0.8423, 0.6968, 0.8218, 0.9834, 0.9894, 0.9974, 0.9999, 1.0000]
    )

    assert_allclose(
        published_objectives[1:] / published_objectives[:-1],
        published_ratios,
        atol=1e-4,
    )
    assert select_change_count(published_objectives, nu=0.05) == 4


def test_constant_signal_has_no_change_points():
    detector = CachalotDetector()

    result = detector.fit_predict(np.full(20, 3.0))

    assert_array_equal(result, np.empty(0, dtype=np.int64))
    assert_array_equal(detector.change_points_, result)


def test_exact_step_stops_when_the_lasso_path_fits_the_signal():
    y = np.zeros(120)
    y[30:] = 2.0
    detector = CachalotDetector(max_candidates=20)

    result = detector.fit_predict(y)

    assert_array_equal(detector.candidate_points_[0], [30])
    assert_array_equal(result, [30])


def test_multivariate_result_is_union_of_independent_channel_results():
    x = np.zeros((120, 2))
    x[30:, 0] = 2.0
    x[70:, 1] = -1.5
    params = {"max_candidates": 20, "nu": 0.01}
    expected = np.unique(
        np.concatenate(
            [
                CachalotDetector(**params).fit_predict(x[:, channel])
                for channel in range(x.shape[1])
            ]
        )
    )

    actual = CachalotDetector(**params).fit_predict(x)

    assert_array_equal(expected, [30, 70])
    assert_array_equal(actual, expected)


def test_multivariate_channel_first_axis_returns_same_union():
    x = np.zeros((120, 2))
    x[30:, 0] = 2.0
    x[70:, 1] = -1.5

    actual = CachalotDetector(max_candidates=20, axis=1).fit_predict(x.T)

    assert_array_equal(actual, [30, 70])


def test_lasso_candidates_are_invariant_to_signal_units():
    """A small physical unit must not trigger an absolute endpoint tolerance."""
    y = np.repeat([0.0, 2.0, -1.0, 3.0], 30)
    expected = lasso_candidate_points(y, 20)
    for scale in (1e-14, 1e-8, 1e8, 1e14):
        assert_array_equal(lasso_candidate_points(y * scale, 20), expected)
        assert_array_equal(CachalotDetector().fit_predict(y * scale), [30, 60, 90])


def test_non_binary_constant_has_no_candidates():
    """Mean-rounding error must not turn a constant into a Lasso candidate."""
    assert_array_equal(lasso_candidate_points(np.full(120, 0.1), 20), [])


def test_exact_symmetric_steps_preserve_tied_entries():
    y = np.repeat([0.0, 1.0, 0.0], 40)
    assert_array_equal(np.sort(lasso_candidate_points(y, 20)), [40, 80])
    assert_array_equal(CachalotDetector().fit_predict(y), [40, 80])


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("limit", [1, 5, 20, 31])
def test_structured_lars_matches_dense_reference_across_paths(seed, limit):
    y = np.random.default_rng(seed).normal(size=32)
    assert_array_equal(
        lasso_candidate_points(y, limit), _dense_lars_candidates(y, limit)
    )


@pytest.mark.parametrize(
    "seed, expected",
    [
        (0, [29, 49, 66, 70, 90]),
        (1, [29, 49, 69, 89]),
    ],
)
def test_seeded_published_toy_design(seed, expected):
    """Section 3's design, with our own noise seed and zero-based indices.

    This is a synthetic regression, not a reproduction of Table 1's
    unpublished noise realization. Paper variable beta_30 starts sample
    30 (one-based), hence tsseg split index 29.
    """
    beta = np.zeros(100)
    beta[[29, 49, 69, 89]] = [5, -3, 4, -2]
    y = np.cumsum(beta) + np.random.default_rng(seed).normal(size=100)
    detector = CachalotDetector(max_candidates=9, nu=0.05)
    assert_array_equal(detector.fit_predict(y), expected)


@pytest.mark.parametrize("length", [2, 3, 8])
def test_candidate_budget_is_capped_by_series_length(length):
    y = np.arange(length, dtype=float)
    detector = CachalotDetector(max_candidates=50)
    result = detector.fit_predict(y)
    assert len(detector.candidate_points_[0]) <= length - 1
    assert np.all((result > 0) & (result < length))


@pytest.mark.parametrize(
    "params",
    [
        {"max_candidates": 0},
        {"max_candidates": 1.5},
        {"nu": 0.0},
        {"nu": 1.0},
        {"nu": np.nan},
    ],
)
def test_invalid_parameters_are_rejected(params):
    with pytest.raises((ValueError, TypeError)):
        CachalotDetector(**params).fit_predict(np.arange(12.0))
