"""Unit tests for ``tsseg.metrics.BidirectionalCovering``.

The ground-truth direction must equal the classical ``Covering`` score, the
prediction direction is the same computation with the roles swapped, and the
four aggregation strategies combine the two.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tsseg.metrics import BidirectionalCovering, Covering
from tsseg.metrics.bidirectional_covering import _AggregationStrategy

AGGREGATIONS = ["harmonic", "geometric", "arithmetic", "min"]

# Series of length 10, true change point 5, predicted change points 2, 5, 8.
# Ground truth direction: [0, 5) and [5, 10) each best match a predicted
# segment of length 3 (IoU 3/5): 0.6. Prediction direction: the four predicted
# segments have IoU 2/5, 3/5, 3/5, 2/5 with their best true segment:
# (2 * 0.4 + 3 * 0.6 + 3 * 0.6 + 2 * 0.4) / 10 = 0.52.
N = 10
TRUE_CPS = [5]
PRED_CPS = [2, 5, 8]
GT_COVERING = 0.6
PRED_COVERING = 0.52
EXPECTED_SCORES = {
    "harmonic": 2 * 0.6 * 0.52 / (0.6 + 0.52),
    "geometric": math.sqrt(0.6 * 0.52),
    "arithmetic": 0.5 * (0.6 + 0.52),
    "min": 0.52,
}


def _random_change_points(rng, n, max_cps=6):
    k = int(rng.integers(0, max_cps))
    return sorted(rng.choice(np.arange(1, n), size=k, replace=False).tolist())


class TestBidirectionalCovering:
    @pytest.mark.parametrize("aggregation", AGGREGATIONS)
    def test_hand_computed_scores(self, aggregation):
        result = BidirectionalCovering(aggregation=aggregation).compute(
            TRUE_CPS, PRED_CPS, n_timepoints=N
        )
        assert result["ground_truth_covering"] == pytest.approx(GT_COVERING)
        assert result["prediction_covering"] == pytest.approx(PRED_COVERING)
        assert result["score"] == pytest.approx(EXPECTED_SCORES[aggregation])

    def test_ground_truth_direction_equals_covering(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            n = int(rng.integers(5, 300))
            y_true = _random_change_points(rng, n)
            y_pred = _random_change_points(rng, n)
            result = BidirectionalCovering().compute(y_true, y_pred, n_timepoints=n)
            covering = Covering().compute(y_true, y_pred, n_timepoints=n)["score"]
            assert result["ground_truth_covering"] == pytest.approx(covering)

    @pytest.mark.parametrize("aggregation", AGGREGATIONS)
    def test_swapping_inputs_swaps_directions(self, aggregation):
        rng = np.random.default_rng(1)
        metric = BidirectionalCovering(aggregation=aggregation)
        for _ in range(30):
            n = int(rng.integers(5, 300))
            a = _random_change_points(rng, n)
            b = _random_change_points(rng, n)
            forward = metric.compute(a, b, n_timepoints=n)
            backward = metric.compute(b, a, n_timepoints=n)
            assert forward["ground_truth_covering"] == pytest.approx(
                backward["prediction_covering"]
            )
            assert forward["score"] == pytest.approx(backward["score"])

    def test_perfect_and_no_change_points(self):
        for y_true, y_pred in [([3, 7], [3, 7]), ([], [])]:
            result = BidirectionalCovering().compute(y_true, y_pred, n_timepoints=N)
            assert result == {
                "score": 1.0,
                "ground_truth_covering": 1.0,
                "prediction_covering": 1.0,
            }

    def test_numpy_and_list_inputs_agree(self):
        from_lists = BidirectionalCovering().compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
        from_arrays = BidirectionalCovering().compute(
            np.array(TRUE_CPS), np.array(PRED_CPS), n_timepoints=N
        )
        assert from_arrays == pytest.approx(from_lists)

    def test_label_sequences(self):
        y_true = np.repeat([0, 1], 5)
        y_pred = np.repeat([0, 1, 2, 3], [2, 3, 3, 2])
        result = BidirectionalCovering(convert_labels_to_segments=True).compute(
            y_true, y_pred
        )
        assert result["ground_truth_covering"] == pytest.approx(GT_COVERING)
        assert result["prediction_covering"] == pytest.approx(PRED_COVERING)

    def test_aggregation_name_is_normalised(self):
        metric = BidirectionalCovering(aggregation="  MIN ")
        result = metric.compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
        assert result["score"] == pytest.approx(EXPECTED_SCORES["min"])

    def test_unknown_aggregation(self):
        with pytest.raises(ValueError, match="aggregation must be one of"):
            BidirectionalCovering(aggregation="median")


class TestAggregationStrategy:
    @pytest.mark.parametrize("name", ["harmonic", "geometric"])
    def test_zero_coverage_gives_zero(self, name):
        strategy = _AggregationStrategy(name)
        assert strategy(0.0, 0.8) == 0.0
        assert strategy(0.8, 0.0) == 0.0

    def test_arithmetic_and_min_with_zero(self):
        assert _AggregationStrategy("arithmetic")(0.0, 0.8) == pytest.approx(0.4)
        assert _AggregationStrategy("min")(0.0, 0.8) == 0.0

    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            _AggregationStrategy("median")(0.5, 0.5)


class TestSegmentHelpers:
    def test_compute_segments_sorts_and_drops_empty_segments(self):
        segments = BidirectionalCovering._compute_segments([10, 0, 5, 5])
        assert segments == [(0, 5), (5, 10)]
        assert BidirectionalCovering._compute_segments([]) == []

    def test_directional_covering_edge_cases(self):
        covering = BidirectionalCovering._directional_covering
        assert covering([], []) == 1.0
        assert covering([], [(0, 10)]) == 0.0
        assert covering([(0, 10)], []) == 0.0
        assert covering([(3, 3)], [(0, 10)]) == 0.0

    def test_directional_covering_sorts_targets(self):
        covering = BidirectionalCovering._directional_covering
        source = [(0, 5), (5, 10)]
        assert covering(source, [(5, 8), (0, 2), (8, 10), (2, 5)]) == pytest.approx(
            GT_COVERING
        )
