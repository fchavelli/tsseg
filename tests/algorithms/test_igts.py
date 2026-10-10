"""Unit tests for the IGTS helpers and ``InformationGainDetector`` edge cases.

The expected values are computed by hand from the definitions of the IGTS
paper (Sadri et al., 2017): entropy of the channel proportions (Eq. 3-4),
information gain of a segmentation (Eq. 2) and the normalise + complement
augmentation (Eq. 12-13).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import InformationGainDetector
from tsseg.algorithms.igts.detector import (
    _IGTS,
    ChangePointResult,
    _augment_univariate,
    _entropy_from_sums,
    entropy,
    generate_segments,
    generate_segments_pandas,
)


def _cumsum(X: np.ndarray) -> np.ndarray:
    """Cumulative sums with a leading row of zeros, as ``find_change_points``."""
    cumsum = np.zeros((X.shape[0] + 1, X.shape[1]))
    np.cumsum(X, axis=0, out=cumsum[1:])
    return cumsum


def _two_blocks(n_left: int = 5, n_right: int = 5) -> np.ndarray:
    """Two segments with disjoint channel supports: IG of the split = log 2."""
    return np.vstack(
        [np.tile([1.0, 0.0], (n_left, 1)), np.tile([0.0, 1.0], (n_right, 1))]
    )


# ── _augment_univariate (Eq. 12-13) ──────────────────────────────────────


def test_augment_normalizes_each_channel_and_appends_its_complement():
    X = np.array([[1.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    X_aug = _augment_univariate(X)
    # channel 0: shifted [0, 1, 3], sum 4 -> [0, .25, .75]; complement max - c
    # channel 1 is constant: shifted to zeros, kept at zero, complement zero
    expected = np.array(
        [
            [0.0, 0.0, 0.75, 0.0],
            [0.25, 0.0, 0.5, 0.0],
            [0.75, 0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(X_aug, expected)


def test_augment_casts_integer_input_to_float():
    X_aug = _augment_univariate(np.array([[0], [1], [3]]))
    assert X_aug.dtype == np.float64
    np.testing.assert_allclose(X_aug[:, 0], [0.0, 0.25, 0.75])
    np.testing.assert_allclose(X_aug[:, 1], [0.75, 0.5, 0.0])


# ── entropy (Eq. 3-4) and its cumulative-sum version ─────────────────────


def test_entropy_of_equal_channel_totals_is_log_of_channel_count():
    assert entropy(np.ones((5, 4))) == pytest.approx(np.log(4))


def test_entropy_of_a_single_channel_is_zero():
    assert entropy(np.array([[1.0], [2.0]])) == pytest.approx(0.0)


def test_entropy_of_an_empty_segment_is_zero():
    assert entropy(np.empty((0, 3))) == 0.0


def test_entropy_shifts_channels_when_the_total_is_not_positive():
    X = np.array([[-1.0, -3.0], [-2.0, -1.0]])
    # total -7 <= 0: each channel is shifted by its minimum -> [[1, 0], [0, 2]]
    p = np.array([1.0, 2.0]) / 3.0
    assert entropy(X) == pytest.approx(-np.sum(p * np.log(p)))


def test_entropy_of_constant_non_positive_data_is_zero():
    assert entropy(-np.ones((3, 2))) == 0.0


def test_entropy_from_sums_matches_entropy():
    rng = np.random.default_rng(0)
    X = rng.random((20, 3))
    assert _entropy_from_sums(X.sum(axis=0)) == pytest.approx(entropy(X))


def test_entropy_from_sums_ignores_empty_channels():
    assert _entropy_from_sums(np.array([2.0, 0.0, 2.0])) == pytest.approx(np.log(2))


def test_entropy_from_sums_of_zero_sums_is_zero():
    assert _entropy_from_sums(np.zeros(3)) == 0.0


# ── segment generators ───────────────────────────────────────────────────


def test_generate_segments_splits_at_the_boundaries():
    X = np.arange(10).reshape(-1, 1)
    segments = list(generate_segments(X, [0, 3, 7, 10]))
    assert [s.ravel().tolist() for s in segments] == [
        [0, 1, 2],
        [3, 4, 5, 6],
        [7, 8, 9],
    ]


def test_generate_segments_pandas_matches_and_sorts_the_boundaries():
    X = np.arange(10).reshape(-1, 1)
    expected = [s.ravel().tolist() for s in generate_segments(X, [0, 3, 7, 10])]
    segments = generate_segments_pandas(X, [7, 0, 10, 3])
    assert [s.ravel().tolist() for s in segments] == expected


# ── information gain (Eq. 2) ─────────────────────────────────────────────


def test_information_gain_of_the_identity_segmentation_is_zero():
    X = _two_blocks()
    assert _IGTS.information_gain_score(X, [0, 10]) == pytest.approx(0.0)


def test_information_gain_of_a_pure_split_is_the_total_entropy():
    X = _two_blocks()
    assert _IGTS.information_gain_score(X, [0, 5, 10]) == pytest.approx(np.log(2))
    ig = _IGTS._ig_from_cumsum(_cumsum(X), 10, np.log(2), [0, 5, 10])
    assert ig == pytest.approx(np.log(2))


@pytest.mark.parametrize("change_points", [[0, 30], [0, 7, 30], [0, 4, 11, 29, 30]])
def test_information_gain_from_cumsum_matches_the_direct_formula(change_points):
    rng = np.random.default_rng(1)
    X = rng.random((30, 4))
    h_total = _entropy_from_sums(X.sum(axis=0))
    ig = _IGTS._ig_from_cumsum(_cumsum(X), 30, h_total, change_points)
    assert ig == pytest.approx(_IGTS.information_gain_score(X, change_points))


def test_information_gain_from_cumsum_skips_empty_segments():
    X = _two_blocks()
    cumsum = _cumsum(X)
    with_empty = _IGTS._ig_from_cumsum(cumsum, 10, np.log(2), [0, 5, 5, 10])
    without = _IGTS._ig_from_cumsum(cumsum, 10, np.log(2), [0, 5, 10])
    assert with_empty == pytest.approx(without)


# ── _IGTS search ─────────────────────────────────────────────────────────


def test_identity_returns_the_series_boundaries():
    assert _IGTS().identity(np.zeros((12, 2))) == [0, 12]


def test_candidates_are_the_step_grid_without_existing_change_points():
    assert _IGTS(step=3).get_candidates(10, [0, 6, 10]) == [3, 9]


def test_find_change_points_places_the_split_between_pure_blocks():
    igts = _IGTS(k_max=1, step=1)
    assert igts.find_change_points(_two_blocks()) == [0, 5, 10]
    assert igts.intermediate_results_ == [
        ChangePointResult(k=0, score=pytest.approx(np.log(2)), change_points=[0, 5, 10])
    ]


def test_find_change_points_stops_when_the_candidates_run_out():
    # step=5 on 30 points leaves 5 candidates (5, 10, ..., 25) for k_max=10
    X = np.random.default_rng(2).random((30, 2))
    igts = _IGTS(k_max=10, step=5)
    assert igts.find_change_points(X) == [0, 5, 10, 15, 20, 25, 30]
    assert len(igts.intermediate_results_) == 5


# ── InformationGainDetector ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("k_max", "step", "expected"),
    [(10, 5, [5, 10, 15, 20, 25]), (3, 10, [10, 20]), (2, 15, [15])],
)
def test_detector_returns_valid_change_points_when_k_max_exceeds_candidates(
    k_max, step, expected
):
    X = np.r_[np.zeros(15), np.ones(15)] + np.random.default_rng(0).normal(0, 0.1, 30)
    cps = InformationGainDetector(k_max=k_max, step=step).fit_predict(X)
    assert cps.tolist() == expected


def test_detector_recovers_a_noise_free_mean_shift():
    X = np.r_[np.zeros(50), np.ones(50)]
    cps = InformationGainDetector(k_max=1, step=5).fit_predict(X)
    assert cps.tolist() == [50]


def test_detector_returns_exactly_k_max_change_points_on_the_step_grid():
    X = np.random.default_rng(3).random((200, 3))
    detector = InformationGainDetector(k_max=4, step=7)
    cps = detector.fit_predict(X)
    assert cps.dtype.kind == "i"
    assert len(cps) == 4
    assert np.all(np.diff(cps) > 0)
    assert np.all((cps > 0) & (cps < 200))
    assert np.all(cps % 7 == 0)


def test_detector_information_gain_never_decreases_along_the_search():
    X = np.random.default_rng(4).random((120, 2))
    detector = InformationGainDetector(k_max=6, step=3).fit(X)
    detector.predict(X)
    scores = [result.score for result in detector.intermediate_results_]
    assert [result.k for result in detector.intermediate_results_] == list(range(6))
    assert np.all(np.diff(scores) >= -1e-12)


def test_detector_with_k_max_zero_returns_no_change_point():
    detector = InformationGainDetector(k_max=0)
    cps = detector.fit_predict(np.random.default_rng(5).random(50))
    assert cps.dtype.kind == "i"
    assert cps.size == 0
    assert detector.intermediate_results_ == []


def test_detector_repr_shows_the_search_parameters():
    text = repr(InformationGainDetector(k_max=3, step=2))
    assert "k_max=3" in text
    assert "step=2" in text


def test_detector_test_params_run():
    params = InformationGainDetector._get_test_params()
    cps = InformationGainDetector(**params).fit_predict(_two_blocks(10, 10))
    # the pure split comes first; once both blocks are pure, every further
    # split ties at IG = log 2 (up to rounding), so only its count is fixed
    assert 10 in cps.tolist()
    assert len(cps) == 2
