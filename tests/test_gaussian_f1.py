"""Unit tests of ``GaussianF1Score``.

Every true change point carries a Gaussian reward
``exp(-0.5 * (d / sigma) ** 2)`` for a prediction ``d`` samples away, with
``sigma = max(min_sigma, sigma_fraction * n)`` (or, with ``adaptive_sigma``,
half the mean gap to the neighbouring true change points). Predictions are
paired greedily, by decreasing reward; precision and recall divide the summed
reward of the pairs by the number of predicted and true change points.
Expected values below are computed by hand from these definitions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsseg.metrics import GaussianF1Score

N = 1000


def _reward(distance: float, sigma: float) -> float:
    return float(np.exp(-0.5 * (distance / sigma) ** 2))


def _scores(metric: GaussianF1Score, y_true, y_pred, n: int = N) -> dict:
    return metric.compute(y_true, y_pred, n_timepoints=n)


class TestRewardAndWidth:
    def test_exact_prediction_scores_one(self):
        result = _scores(GaussianF1Score(), [250, 500, 750], [250, 500, 750])
        assert result == {
            "score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "matched_weight": 3.0,
        }

    @pytest.mark.parametrize("offset", [-25, -10, -3, 1, 10, 25])
    def test_reward_decays_as_a_gaussian_of_the_distance(self, offset):
        # sigma = 0.01 * 1000 = 10 samples.
        result = _scores(GaussianF1Score(), [500], [500 + offset])
        expected = _reward(offset, 10.0)
        assert result["matched_weight"] == pytest.approx(expected)
        assert result["precision"] == pytest.approx(expected)
        assert result["recall"] == pytest.approx(expected)
        assert result["score"] == pytest.approx(expected)

    def test_width_is_a_fraction_of_the_series_length(self):
        # sigma = 0.05 * 1000 = 50 samples: one sigma away gives exp(-1/2).
        result = _scores(GaussianF1Score(sigma_fraction=0.05), [400], [450])
        assert result["score"] == pytest.approx(np.exp(-0.5))

    def test_width_is_clamped_below_by_min_sigma(self):
        # 0.01 * 50 = 0.5 < min_sigma = 1, so sigma = 1.
        result = _scores(GaussianF1Score(), [20], [21], n=50)
        assert result["score"] == pytest.approx(np.exp(-0.5))
        # A larger min_sigma widens the reward.
        wide = _scores(GaussianF1Score(min_sigma=4.0), [20], [24], n=50)
        assert wide["score"] == pytest.approx(np.exp(-0.5))

    def test_same_width_for_every_change_point(self):
        # Both true change points share sigma = 10, whatever their spacing.
        result = _scores(GaussianF1Score(), [100, 900], [110, 880])
        weight = _reward(10, 10.0) + _reward(20, 10.0)
        assert result["matched_weight"] == pytest.approx(weight)
        assert result["score"] == pytest.approx(weight / 2)

    def test_far_predictions_score_zero(self):
        # 800 samples at sigma = 10: the reward underflows to 0.
        result = _scores(GaussianF1Score(), [100], [900])
        assert result == {
            "score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "matched_weight": 0.0,
        }


class TestPrecisionRecall:
    def test_missed_change_points_lower_recall_only(self):
        result = _scores(GaussianF1Score(), [250, 500, 750], [500])
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1 / 3)
        assert result["score"] == pytest.approx(0.5)

    def test_spurious_predictions_lower_precision_only(self):
        result = _scores(GaussianF1Score(), [500], [200, 500, 800])
        assert result["precision"] == pytest.approx(1 / 3)
        assert result["recall"] == pytest.approx(1.0)
        assert result["score"] == pytest.approx(0.5)

    def test_score_is_the_harmonic_mean(self):
        result = _scores(GaussianF1Score(), [250, 500, 750], [505, 760])
        p, r = result["precision"], result["recall"]
        assert result["score"] == pytest.approx(2 * p * r / (p + r))
        weight = _reward(5, 10.0) + _reward(10, 10.0)
        assert p == pytest.approx(weight / 2)
        assert r == pytest.approx(weight / 3)

    def test_a_prediction_is_paired_with_one_true_change_point_only(self):
        # One prediction between two true change points earns one reward.
        result = _scores(GaussianF1Score(), [495, 505], [500])
        assert result["matched_weight"] == pytest.approx(_reward(5, 10.0))
        assert result["recall"] == pytest.approx(_reward(5, 10.0) / 2)

    def test_duplicated_predictions_count_as_spurious(self):
        # Same convention as F1Score: the copy finds no free true change point.
        result = _scores(GaussianF1Score(), [500], [500, 500])
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(1.0)


class TestGreedyMatching:
    def test_pairs_are_taken_by_decreasing_reward(self):
        # sigma = 4. The closest pair (104, 103) is taken first, which leaves
        # (100, 107) to the other prediction: 0.969 + 0.216 = 1.185.
        result = _scores(GaussianF1Score(sigma_fraction=0.004), [100, 104], [103, 107])
        greedy = _reward(1, 4.0) + _reward(7, 4.0)
        assert result["matched_weight"] == pytest.approx(greedy)

    def test_greedy_pairing_can_be_below_the_optimal_assignment(self):
        # The assignment (100, 103), (104, 107) would earn 2 * 0.755 = 1.510:
        # the greedy pairing is not a maximum-weight matching.
        result = _scores(GaussianF1Score(sigma_fraction=0.004), [100, 104], [103, 107])
        optimal = 2 * _reward(3, 4.0)
        assert result["matched_weight"] < optimal


class TestAdaptiveSigma:
    def test_width_is_half_the_mean_gap_to_the_neighbours(self):
        # Gaps: 300 -> 400 is 100 on both sides, so sigma = 50 for both.
        metric = GaussianF1Score(adaptive_sigma=True)
        result = _scores(metric, [300, 400], [330, 400])
        weight = _reward(30, 50.0) + 1.0
        assert result["matched_weight"] == pytest.approx(weight)
        assert result["score"] == pytest.approx(weight / 2)

    @pytest.mark.parametrize(
        "y_pred",
        [
            # sigma(100) = 0.5 * 200 = 100
            [200, 300, 900],
            # sigma(300) = 0.5 * (200 + 600) / 2 = 200
            [100, 500, 900],
            # sigma(900) = 0.5 * 600 = 300
            [100, 300, 600],
        ],
    )
    def test_width_differs_per_change_point(self, y_pred):
        # Two exact predictions (reward 1, paired first) and one prediction
        # one sigma away from the remaining true change point.
        metric = GaussianF1Score(adaptive_sigma=True)
        result = _scores(metric, [100, 300, 900], y_pred)
        assert result["matched_weight"] == pytest.approx(2 + np.exp(-0.5))

    def test_single_change_point_uses_a_tenth_of_the_series(self):
        metric = GaussianF1Score(adaptive_sigma=True)
        result = _scores(metric, [500], [600])
        assert result["score"] == pytest.approx(np.exp(-0.5))


class TestDegenerateCases:
    def test_no_change_point_at_all_is_perfect(self):
        result = _scores(GaussianF1Score(), [], [], n=100)
        assert result["score"] == 1.0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_predictions_on_a_stationary_series_score_zero(self):
        assert _scores(GaussianF1Score(), [], [50], n=100)["score"] == 0.0

    def test_no_prediction_scores_zero(self):
        assert _scores(GaussianF1Score(), [50], [], n=100)["score"] == 0.0

    def test_boundaries_are_ignored(self):
        with_bounds = _scores(GaussianF1Score(), [0, 500, N], [0, 510, N])
        without = _scores(GaussianF1Score(), [500], [510])
        assert with_bounds == without


class TestInputs:
    def test_arrays_lists_and_order_are_equivalent(self):
        expected = _scores(GaussianF1Score(), [250, 750], [260, 740])
        as_arrays = _scores(
            GaussianF1Score(), np.array([750, 250]), np.array([740, 260])
        )
        assert as_arrays == expected

    def test_call_matches_compute(self):
        metric = GaussianF1Score()
        assert metric([500], [510], n_timepoints=N) == _scores(metric, [500], [510])

    def test_label_sequences(self):
        # Labels change at 50 (true) and 52 (predicted); n = 100, sigma = 1.
        y_true = np.repeat([0, 1], 50)
        y_pred = np.repeat([0, 1], [52, 48])
        metric = GaussianF1Score(convert_labels_to_segments=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = metric.compute(y_true, y_pred)
        assert result["score"] == pytest.approx(_reward(2, 1.0))

    def test_without_series_length_the_last_true_point_is_the_length(self):
        # Documented convention: y_true must then end with n.
        metric = GaussianF1Score()
        with pytest.warns(UserWarning, match="series length is inferred"):
            result = metric.compute([500, N], [510])
        assert result["score"] == pytest.approx(_reward(10, 10.0))


@pytest.mark.parametrize("seed", range(5))
def test_scores_stay_in_the_unit_interval(seed):
    rng = np.random.default_rng(seed)
    y_true = np.sort(
        rng.choice(np.arange(1, N), size=rng.integers(1, 8), replace=False)
    )
    y_pred = np.sort(
        rng.choice(np.arange(1, N), size=rng.integers(1, 8), replace=False)
    )
    for metric in (GaussianF1Score(), GaussianF1Score(adaptive_sigma=True)):
        result = _scores(metric, y_true, y_pred)
        for key in ("score", "precision", "recall"):
            assert 0.0 <= result[key] <= 1.0
        assert result["matched_weight"] <= min(len(y_true), len(y_pred))
