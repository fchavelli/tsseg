"""Unit tests for the state-detection metrics (``tsseg.metrics.state_detection``).

Covers the scikit-learn wrappers (ARI, NMI, AMI), the State Matching Score
(SMS, one hand-computed case per error type) and the boundary-weighted WARI
and WNMI, including their reduction to ARI and NMI when every point has the
same weight.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix

from tsseg.metrics import (
    AdjustedMutualInformation,
    AdjustedRandIndex,
    NormalizedMutualInformation,
    StateMatchingScore,
    WeightedAdjustedRandIndex,
    WeightedNormalizedMutualInformation,
)
from tsseg.metrics.state_detection import (
    _compute_boundary_distances,
    _map_predicted_labels,
    _normalized_block_boundary_distance,
    weighted_adjusted_rand_score,
    weighted_contingency_matrix,
    weighted_entropy,
    weighted_mutual_info_score,
    weighted_normalized_mutual_info_score,
    weighted_pair_confusion_matrix,
)

AVERAGE_METHODS = ["min", "geometric", "arithmetic", "max"]


def _piecewise_labels(rng, n, n_segments, n_states=4):
    """Random piecewise-constant state labels of length ``n``."""
    labels = np.zeros(n, dtype=int)
    cps = np.sort(rng.choice(np.arange(1, n), size=n_segments - 1, replace=False))
    for cp in cps:
        labels[cp:] = rng.integers(0, n_states)
    return labels


def _random_pairs(seed, n_pairs=25, min_len=10, max_len=200):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n_pairs):
        n = int(rng.integers(min_len, max_len))
        y_true = _piecewise_labels(rng, n, int(rng.integers(1, 6)))
        y_pred = _piecewise_labels(rng, n, int(rng.integers(1, 6)))
        pairs.append((y_true, y_pred))
    return pairs


def _relabel(labels, offset=10):
    """Rename the labels (reversed order plus an offset): same partition."""
    uniques = np.unique(labels)
    mapping = {u: offset + len(uniques) - i for i, u in enumerate(uniques)}
    return np.array([mapping[v] for v in labels])


def _ari_from_weighted_counts(y_true, y_pred, weights):
    """ARI formula (Hubert & Arabie) with weighted counts n_ij = sum of w_k.

    The WARI definition substitutes the weighted contingency counts into the
    ARI formula, with C(x, 2) = x (x - 1) / 2 for real x.
    """
    table = weighted_contingency_matrix(y_true, y_pred, weights)

    def pairs(x):
        return x * (x - 1.0) / 2.0

    index = pairs(table).sum()
    rows = pairs(table.sum(axis=1)).sum()
    cols = pairs(table.sum(axis=0)).sum()
    expected = rows * cols / pairs(table.sum())
    maximum = 0.5 * (rows + cols)
    return (index - expected) / (maximum - expected)


# Two states of 50 points; three mislabelled points either right after the
# true change point (``near``) or in the middle of the first segment
# (``far``). Both predictions have the same unweighted contingency table.
Y_TWO_STATES = np.repeat([0, 1], 50)
Y_ERRORS_NEAR = Y_TWO_STATES.copy()
Y_ERRORS_NEAR[50:53] = 0
Y_ERRORS_FAR = Y_TWO_STATES.copy()
Y_ERRORS_FAR[20:23] = 1


# ---------------------------------------------------------------------------
# scikit-learn wrappers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_cls", "reference"),
    [
        (AdjustedRandIndex, adjusted_rand_score),
        (NormalizedMutualInformation, normalized_mutual_info_score),
        (AdjustedMutualInformation, adjusted_mutual_info_score),
    ],
)
class TestSklearnWrappers:
    def test_matches_sklearn(self, metric_cls, reference):
        for y_true, y_pred in _random_pairs(seed=0):
            result = metric_cls().compute(y_true, y_pred)
            assert set(result) == {"score"}
            assert result["score"] == pytest.approx(reference(y_true, y_pred))

    def test_invariant_to_label_names(self, metric_cls, reference):
        for y_true, y_pred in _random_pairs(seed=1, n_pairs=10):
            metric = metric_cls()
            assert metric(y_true, _relabel(y_pred))["score"] == pytest.approx(
                metric(y_true, y_pred)["score"]
            )

    def test_perfect(self, metric_cls, reference):
        y = np.repeat([0, 1, 2], 10)
        assert metric_cls().compute(y, _relabel(y))["score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Label mapping and boundary helpers (used by SMS and WARI/WNMI)
# ---------------------------------------------------------------------------


class TestMapPredictedLabels:
    def test_renames_predicted_labels_after_the_true_ones(self):
        mapped = _map_predicted_labels([0, 0, 1, 1, 2, 2], [7, 7, 3, 3, 5, 5])
        np.testing.assert_array_equal(mapped, [0, 0, 1, 1, 2, 2])

    def test_keeps_one_output_label_per_predicted_label(self):
        # Three predicted labels, one true label: the best overlap takes the
        # true label, the others keep their own (free) value.
        mapped = _map_predicted_labels([0, 0, 0, 0], [0, 0, 1, 2])
        np.testing.assert_array_equal(mapped, [0, 0, 1, 2])

    def test_taken_value_falls_back_to_smallest_free_integer(self):
        # Predicted 0 takes the true label 5; predicted 5 cannot keep 5.
        mapped = _map_predicted_labels([5, 5, 5, 5], [0, 0, 5, 1])
        np.testing.assert_array_equal(mapped, [5, 5, 0, 1])

    def test_only_the_common_prefix_is_matched(self):
        mapped = _map_predicted_labels([0, 0, 1, 1], [5, 5, 6, 6, 6, 7])
        np.testing.assert_array_equal(mapped, [0, 0, 1, 1, 1, 7])

    def test_empty_inputs(self):
        assert _map_predicted_labels([0, 1], []).size == 0
        # Without true labels, predicted labels are renumbered in sorted order.
        np.testing.assert_array_equal(_map_predicted_labels([], [3, 3, 1]), [1, 1, 0])

    def test_preserves_shape(self):
        mapped = _map_predicted_labels(np.array([0, 0, 1, 1]), np.array([[4, 4, 9, 9]]))
        assert mapped.shape == (1, 4)
        np.testing.assert_array_equal(mapped, [[0, 0, 1, 1]])


class TestBoundaryDistances:
    def test_distance_to_nearest_boundary_including_both_ends(self):
        # Boundaries at 0, 3 (label change) and 5 (series length).
        distances = _compute_boundary_distances(np.array([0, 0, 0, 1, 1]))
        np.testing.assert_array_equal(distances, [0, 1, 1, 0, 1])

    def test_constant_labels_get_distance_n(self):
        np.testing.assert_array_equal(
            _compute_boundary_distances(np.full(4, 2)), [4] * 4
        )

    def test_block_distance_is_normalised_by_half_the_length(self):
        # Block [5, 5] between boundaries 0 and 10: gap 5, d = 2 * 5 / 20.
        assert _normalized_block_boundary_distance([0, 10, 20], 5, 5, 20) == 0.5
        # Block [12, 13]: gaps 2 (to 10) and 7 (to 20).
        assert _normalized_block_boundary_distance([0, 10, 20], 12, 13, 20) == 0.2

    def test_block_starting_on_a_boundary_has_zero_distance(self):
        assert _normalized_block_boundary_distance([0, 10, 20], 10, 12, 20) == 0.0

    def test_degenerate_inputs(self):
        assert _normalized_block_boundary_distance([], 3, 4, 20) == 0.0
        assert _normalized_block_boundary_distance([0, 10], 3, 4, 0) == 0.0


# ---------------------------------------------------------------------------
# State Matching Score
# ---------------------------------------------------------------------------

# Default weights: delay 0.1, transition 0.3, isolation 0.8, missing 0.5.
# SMS = 1 - sum over error blocks of l * (1 + extra) / n, with extra =
# w_delay (delay), d * w_e (isolation, transition; d = 2 * gap / n to the
# nearest true change point around the block) or
# w_m * (1 + 3 * (w_m - 1) / A) (missing, A = number of true runs in the block).
SMS_TRUE = np.repeat([0, 1], 10)  # n = 20, true change point at 10
SMS_CASES = {
    # name: (y_true, y_pred, expected errors (type, start, end, penalty), score)
    "delay": (
        SMS_TRUE,
        np.repeat([0, 1], [12, 8]),
        [("delay", 10, 11, 2 * 0.1)],
        1 - (2 + 0.2) / 20,
    ),
    "isolation": (
        SMS_TRUE,
        np.where(np.arange(20) == 5, 1, SMS_TRUE),
        [("isolation", 5, 5, 1 * 0.8 * (2 * 5 / 20))],
        1 - (1 + 0.4) / 20,
    ),
    "transition": (
        SMS_TRUE,
        np.repeat([0, 2, 1], [8, 4, 8]),
        # Block [8, 11] straddles the change point 10; the gaps are taken to
        # the change points around the block (0 and 20): min(8, 9) = 8.
        [("transition", 8, 11, 4 * 0.3 * (2 * 8 / 20))],
        1 - (4 + 0.96) / 20,
    ),
    "missing": (
        np.repeat([1, 2, 3, 0], [2, 2, 2, 14]),
        np.zeros(20, dtype=int),
        [("missing", 0, 5, 6 * 0.5 * (1 + 3 * (0.5 - 1) / 3))],
        1 - (6 + 1.5) / 20,
    ),
    "mislabelled_segment_on_a_boundary": (
        SMS_TRUE,
        np.repeat([0, 2, 1], [10, 3, 7]),
        # Isolation block starting on the true change point: d = 0, the block
        # costs its length only.
        [("isolation", 10, 12, 0.0)],
        1 - 3 / 20,
    ),
}


class TestStateMatchingScore:
    @pytest.mark.parametrize("case", sorted(SMS_CASES))
    def test_hand_computed_error_blocks(self, case):
        y_true, y_pred, expected_errors, expected_score = SMS_CASES[case]
        result = StateMatchingScore().compute(y_true, y_pred, return_errors=True)

        assert result["score"] == pytest.approx(expected_score)
        errors = [
            (e["type"], e["start"], e["end"], e["penalty"]) for e in result["errors"]
        ]
        assert len(errors) == len(expected_errors)
        for got, want in zip(errors, expected_errors, strict=True):
            assert got[:3] == want[:3]
            assert got[3] == pytest.approx(want[3])

    def test_perfect_and_relabelled_predictions(self):
        result = StateMatchingScore().compute(
            SMS_TRUE, _relabel(SMS_TRUE), return_errors=True
        )
        assert result["score"] == 1.0
        assert result["errors"] == []

    def test_score_decomposes_into_error_blocks(self):
        for y_true, y_pred in _random_pairs(seed=2):
            result = StateMatchingScore().compute(y_true, y_pred, return_errors=True)
            cost = sum(e["size"] + e["penalty"] for e in result["errors"])
            assert 1.0 - result["score"] == pytest.approx(cost / len(y_true))
            for e in result["errors"]:
                assert e["size"] == e["end"] - e["start"] + 1
                assert e["type"] in {"delay", "isolation", "transition", "missing"}

    def test_custom_weights_and_default_weight_of_one(self):
        y_true, y_pred, _, _ = SMS_CASES["delay"]
        metric = StateMatchingScore(weights={"delay": 0.5})
        assert metric.compute(y_true, y_pred)["score"] == pytest.approx(1 - 3 / 20)

        # An error type missing from ``weights`` gets the weight 1.0.
        y_true, y_pred, _, _ = SMS_CASES["isolation"]
        expected = 1 - (1 + 1.0 * 0.5) / 20
        assert metric.compute(y_true, y_pred)["score"] == pytest.approx(expected)

    def test_weights_are_copied(self):
        weights = {"delay": 0.2}
        metric = StateMatchingScore(weights=weights)
        metric.weights["delay"] = 0.9
        assert weights == {"delay": 0.2}

        default = StateMatchingScore()
        default.weights["delay"] = 0.9
        assert StateMatchingScore.DEFAULT_WEIGHTS["delay"] == 0.1

    @pytest.mark.parametrize(
        ("flags", "keys"),
        [
            ({}, {"score"}),
            ({"return_mapped": True}, {"score", "mapped_pred"}),
            ({"return_errors": True}, {"score", "errors"}),
            (
                {"return_mapped": True, "return_errors": True},
                {"score", "mapped_pred", "errors"},
            ),
        ],
    )
    def test_optional_outputs(self, flags, keys):
        y_true, y_pred, _, expected_score = SMS_CASES["transition"]
        result = StateMatchingScore().compute(y_true, y_pred, **flags)
        assert set(result) == keys
        assert isinstance(result["score"], float)
        assert result["score"] == pytest.approx(expected_score)
        if "mapped_pred" in result:
            np.testing.assert_array_equal(result["mapped_pred"], y_pred)

    def test_empty_input_scores_one(self):
        assert StateMatchingScore().compute(np.array([]), np.array([])) == {
            "score": 1.0
        }


# ---------------------------------------------------------------------------
# Weighted ARI
# ---------------------------------------------------------------------------


class TestWeightedAdjustedRandIndex:
    def test_reduces_to_ari_with_alpha_zero(self):
        for y_true, y_pred in _random_pairs(seed=3):
            score = WeightedAdjustedRandIndex(alpha=0.0).compute(y_true, y_pred)[
                "score"
            ]
            assert score == pytest.approx(
                adjusted_rand_score(y_true, y_pred), abs=1e-12
            )

    def test_custom_distance_function(self):
        metric = WeightedAdjustedRandIndex(distance_func=lambda d: np.ones(len(d)))
        for y_true, y_pred in _random_pairs(seed=4, n_pairs=10):
            assert metric(y_true, y_pred)["score"] == pytest.approx(
                adjusted_rand_score(y_true, y_pred), abs=1e-9
            )

    def test_perfect_and_relabelled(self):
        y = np.repeat([0, 1, 2], [10, 15, 5])
        assert WeightedAdjustedRandIndex().compute(y, _relabel(y))["score"] == 1.0

    def test_invariant_to_predicted_label_names(self):
        for y_true, y_pred in _random_pairs(
            seed=5, n_pairs=10, min_len=300, max_len=600
        ):
            metric = WeightedAdjustedRandIndex()
            assert metric(y_true, _relabel(y_pred))["score"] == pytest.approx(
                metric(y_true, y_pred)["score"], abs=1e-9
            )

    def test_errors_far_from_boundaries_cost_more(self):
        ari_near = adjusted_rand_score(Y_TWO_STATES, Y_ERRORS_NEAR)
        ari_far = adjusted_rand_score(Y_TWO_STATES, Y_ERRORS_FAR)
        assert ari_near == pytest.approx(ari_far)

        metric = WeightedAdjustedRandIndex()
        assert (
            metric(Y_TWO_STATES, Y_ERRORS_NEAR)["score"]
            > metric(Y_TWO_STATES, Y_ERRORS_FAR)["score"]
        )

    def test_pair_confusion_matrix_with_unit_weights_matches_sklearn(self):
        for y_true, y_pred in _random_pairs(seed=6, n_pairs=10):
            weighted = weighted_pair_confusion_matrix(
                y_true, y_pred, np.ones(len(y_true))
            )
            np.testing.assert_allclose(weighted, pair_confusion_matrix(y_true, y_pred))

    def test_agreeing_pairs_use_weighted_counts(self):
        # C[1, 1] = 2 * sum_ij C(n_ij, 2) with n_ij the weighted counts.
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1, 1])
        weights = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0])
        table = weighted_contingency_matrix(y_true, y_pred, weights)
        matrix = weighted_pair_confusion_matrix(y_true, y_pred, weights)
        assert matrix[1, 1] == pytest.approx((table * (table - 1)).sum())
        assert matrix.sum() == pytest.approx(weights.sum() ** 2 - weights.sum())

    def test_long_series_match_the_weighted_count_formula(self):
        rng = np.random.default_rng(7)
        for _ in range(5):
            y_true = _piecewise_labels(rng, 2000, 6)
            y_pred = _piecewise_labels(rng, 2000, 6)
            weights = 1.0 + 0.1 * _compute_boundary_distances(y_true)
            assert weighted_adjusted_rand_score(
                y_true, y_pred, weights
            ) == pytest.approx(
                _ari_from_weighted_counts(y_true, y_pred, weights), abs=1e-6
            )

    @pytest.mark.xfail(
        strict=True,
        reason="T50: the weighted pair counts are truncated with int(), which "
        "shifts WARI on short series (here 0.2005 instead of 0.2012)",
    )
    def test_short_series_match_the_weighted_count_formula(self):
        y_true = np.repeat([0, 1, 2], [7, 6, 7])
        y_pred = np.repeat([0, 1, 2, 1], [5, 6, 4, 5])
        weights = 1.0 + 0.1 * _compute_boundary_distances(y_true)
        assert weighted_adjusted_rand_score(y_true, y_pred, weights) == pytest.approx(
            _ari_from_weighted_counts(y_true, y_pred, weights), abs=1e-12
        )


# ---------------------------------------------------------------------------
# Weighted NMI
# ---------------------------------------------------------------------------


class TestWeightedNormalizedMutualInformation:
    @pytest.mark.parametrize("average_method", AVERAGE_METHODS)
    def test_reduces_to_nmi_with_alpha_zero(self, average_method):
        metric = WeightedNormalizedMutualInformation(
            alpha=0.0, average_method=average_method
        )
        for y_true, y_pred in _random_pairs(seed=8):
            expected = normalized_mutual_info_score(
                y_true, y_pred, average_method=average_method
            )
            assert metric(y_true, y_pred)["score"] == pytest.approx(expected, abs=1e-12)

    def test_mutual_information_with_unit_weights_matches_sklearn(self):
        for y_true, y_pred in _random_pairs(seed=9, n_pairs=10):
            weighted = weighted_mutual_info_score(y_true, y_pred, np.ones(len(y_true)))
            assert weighted == pytest.approx(
                mutual_info_score(y_true, y_pred), abs=1e-12
            )

    def test_invariant_to_predicted_label_names(self):
        for y_true, y_pred in _random_pairs(seed=10, n_pairs=10):
            metric = WeightedNormalizedMutualInformation()
            assert metric(y_true, _relabel(y_pred))["score"] == pytest.approx(
                metric(y_true, y_pred)["score"]
            )

    def test_errors_far_from_boundaries_cost_more(self):
        nmi_near = normalized_mutual_info_score(Y_TWO_STATES, Y_ERRORS_NEAR)
        nmi_far = normalized_mutual_info_score(Y_TWO_STATES, Y_ERRORS_FAR)
        assert nmi_near == pytest.approx(nmi_far)

        metric = WeightedNormalizedMutualInformation()
        assert (
            metric(Y_TWO_STATES, Y_ERRORS_NEAR)["score"]
            > metric(Y_TWO_STATES, Y_ERRORS_FAR)["score"]
        )

    def test_special_cases(self):
        weights = np.ones(6)
        one_state = np.zeros(6, dtype=int)
        two_states = np.repeat([0, 1], 3)
        # A single state on both sides is a perfect match.
        assert (
            weighted_normalized_mutual_info_score(one_state, one_state + 4, weights)
            == 1.0
        )
        # No shared information: score 0 (up to rounding, unlike sklearn,
        # which returns exactly 0 when one labelling has a single cluster).
        assert weighted_normalized_mutual_info_score(
            one_state, two_states, weights
        ) == pytest.approx(0.0, abs=1e-12)
        assert weighted_normalized_mutual_info_score(
            two_states, two_states, weights
        ) == pytest.approx(1.0)

    def test_unknown_average_method(self):
        with pytest.raises(ValueError, match="average_method"):
            WeightedNormalizedMutualInformation(average_method="median").compute(
                np.repeat([0, 1], 5), np.repeat([0, 1], [4, 6])
            )


class TestWeightedHelpers:
    def test_weighted_entropy(self):
        labels = np.array([0, 0, 1, 1])
        assert weighted_entropy(labels, np.ones(4)) == pytest.approx(np.log(2))
        # Weights 1 vs 3 on the two labels: p = (1/4, 3/4).
        expected = -(0.25 * np.log(0.25) + 0.75 * np.log(0.75))
        assert weighted_entropy(
            labels, np.array([0.5, 0.5, 1.5, 1.5])
        ) == pytest.approx(expected)
        assert weighted_entropy(labels, np.zeros(4)) == 0.0

    def test_weighted_contingency_matrix(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([5, 7, 7, 7])
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        dense = weighted_contingency_matrix(y_true, y_pred, weights)
        np.testing.assert_allclose(dense, [[1.0, 2.0], [0.0, 7.0]])

        sparse = weighted_contingency_matrix(y_true, y_pred, weights, sparse=True)
        np.testing.assert_allclose(sparse.toarray(), dense)

        smoothed = weighted_contingency_matrix(y_true, y_pred, weights, eps=0.5)
        np.testing.assert_allclose(smoothed, dense + 0.5)
