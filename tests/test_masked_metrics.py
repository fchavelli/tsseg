"""Metrics on a reference with unscored points (``scored_intervals``)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tsseg.metrics import (
    AdjustedMutualInformation,
    AdjustedRandIndex,
    BidirectionalCovering,
    Covering,
    F1Score,
    NormalizedMutualInformation,
    StateMatchingScore,
    WeightedAdjustedRandIndex,
    WeightedNormalizedMutualInformation,
    scored_islands,
)
from tsseg.metrics._support import _check_intervals
from tsseg.metrics.state_detection import (
    _compute_boundary_distances,
    weighted_adjusted_rand_score,
)

SD_METRICS = [
    StateMatchingScore,
    AdjustedRandIndex,
    AdjustedMutualInformation,
    NormalizedMutualInformation,
    WeightedAdjustedRandIndex,
    WeightedNormalizedMutualInformation,
]
CPD_METRICS = [
    lambda: F1Score(margin=3),
    Covering,
    BidirectionalCovering,
]


def _labels(rng: np.random.Generator, n: int, k: int, n_segments: int) -> np.ndarray:
    edges = np.sort(rng.choice(np.arange(1, n), size=n_segments - 1, replace=False))
    values = rng.integers(0, k, size=n_segments)
    return np.repeat(values, np.diff(np.r_[0, edges, n]))


def _with_gap(y: np.ndarray, at: int, length: int, fill: int = -1) -> np.ndarray:
    return np.r_[y[:at], np.full(length, fill), y[at:]]


def test_scored_islands_are_the_runs_of_scored_labels() -> None:
    assert scored_islands([0, 0, -1, -1, 1, 1, 2]) == [(0, 2), (4, 7)]
    assert scored_islands([-1, 3, 3, 9, 9, -1], unscored=[-1, 9]) == [(1, 3)]
    assert scored_islands([-1, -1]) == []
    assert scored_islands(np.array([1, 2, 3])) == [(0, 3)]


def test_scored_intervals_are_validated_and_touching_ones_merged() -> None:
    assert _check_intervals([(0, 2), (2, 5), (7, 9)], 10) == [(0, 5), (7, 9)]
    for bad in ([], [(3, 3)], [(-1, 2)], [(0, 11)], [(4, 6), (0, 2)], [(0, 5), (4, 8)]):
        with pytest.raises(ValueError):
            _check_intervals(bad, 10)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("metric", SD_METRICS, ids=lambda m: m.__name__)
def test_full_support_gives_the_unmasked_state_score(metric, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = _labels(rng, 300, 4, 9)
    y_pred = _labels(rng, 300, 5, 12)
    plain = metric().compute(y_true, y_pred)
    masked = metric().compute(y_true, y_pred, scored_intervals=[(0, 300)])
    assert masked["score"] == plain["score"]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("make", CPD_METRICS, ids=["F1", "Covering", "BiC"])
def test_full_support_gives_the_unmasked_change_point_score(make, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    y_true = np.sort(rng.choice(np.arange(1, n), size=6, replace=False))
    y_pred = np.sort(rng.choice(np.arange(1, n), size=9, replace=False))
    plain = make().compute(y_true, y_pred, n_timepoints=n)
    masked = make().compute(y_true, y_pred, n_timepoints=n, scored_intervals=[(0, n)])
    for key, value in plain.items():
        assert masked[key] == value


def test_boundary_distances_match_the_pointwise_definition() -> None:
    rng = np.random.default_rng(0)
    y = _labels(rng, 200, 3, 7)
    marks = np.r_[0, np.flatnonzero(np.diff(y)) + 1, len(y)]
    naive = np.array([np.min(np.abs(i - marks)) for i in range(len(y))])
    np.testing.assert_array_equal(_compute_boundary_distances(y), naive)
    np.testing.assert_array_equal(_compute_boundary_distances(np.zeros(5)), 5)


def test_masked_sms_ignores_the_unknown_gap() -> None:
    # The example of the masked-evaluation contract.
    y_true = np.array([0, 0, -1, -1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1])
    sms = StateMatchingScore()
    assert sms.compute(y_true, y_pred)["score"] < 0.7
    out = sms.compute(y_true, y_pred, scored_intervals=scored_islands(y_true))
    assert out["score"] == 1.0
    assert (out["n_scored"], out["n_total"]) == (4, 6)


def test_masked_sms_is_one_whatever_is_predicted_in_the_gaps() -> None:
    rng = np.random.default_rng(1)
    y_true = _with_gap(_labels(rng, 200, 3, 6), 80, 40)
    y_pred = np.where(y_true == -1, rng.integers(0, 9, size=len(y_true)), y_true + 5)
    out = StateMatchingScore().compute(
        y_true, y_pred, scored_intervals=scored_islands(y_true)
    )
    assert out["score"] == 1.0


@pytest.mark.parametrize("metric", SD_METRICS, ids=lambda m: m.__name__)
def test_masked_state_scores_do_not_depend_on_the_gap_length(metric) -> None:
    rng = np.random.default_rng(2)
    base_true = _labels(rng, 240, 3, 8)
    base_pred = _labels(rng, 240, 4, 10)
    scores = []
    for length in (1, 30, 500):
        y_true = _with_gap(base_true, 100, length)
        y_pred = _with_gap(base_pred, 100, length, fill=7)
        out = metric().compute(y_true, y_pred, scored_intervals=scored_islands(y_true))
        scores.append(out["score"])
    np.testing.assert_allclose(scores, scores[0], rtol=0, atol=1e-12)


def test_masked_sms_fits_one_mapping_and_stops_errors_at_the_gap() -> None:
    # The predicted state 5 means 0 in both islands: one global mapping.
    y_true = np.array([0, 0, 0, 1, 1, -1, -1, 0, 0, 1, 1, 1])
    y_pred = np.array([5, 5, 5, 6, 6, 6, 6, 5, 5, 6, 6, 6])
    out = StateMatchingScore().compute(
        y_true, y_pred, scored_intervals=scored_islands(y_true), return_errors=True
    )
    assert out["score"] == 1.0 and out["errors"] == []
    # An error that runs into the gap stops at its edge.
    y_pred = np.array([5, 5, 5, 6, 5, 5, 5, 5, 5, 6, 6, 6])
    out = StateMatchingScore().compute(
        y_true, y_pred, scored_intervals=scored_islands(y_true), return_errors=True
    )
    assert [(e["start"], e["end"]) for e in out["errors"]] == [(4, 4)]


def test_masked_ari_is_the_ari_of_the_scored_points() -> None:
    rng = np.random.default_rng(3)
    y_true = _with_gap(_labels(rng, 300, 3, 8), 120, 50)
    y_pred = _labels(rng, len(y_true), 4, 9)
    mask = y_true != -1
    out = AdjustedRandIndex().compute(
        y_true, y_pred, scored_intervals=scored_islands(y_true)
    )
    assert out["score"] == adjusted_rand_score(y_true[mask], y_pred[mask])


def test_masked_wari_weights_points_by_their_distance_inside_the_island() -> None:
    y_true = np.array([0, 0, 0, 0, -1, -1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    # Distances to the island edges and true changes: (0, 4) and (6, 10).
    distances = np.array([0, 1, 2, 1, 0, 1, 0, 1], dtype=float)
    mask = y_true != -1
    expected = weighted_adjusted_rand_score(
        y_true[mask], y_pred[mask], 1 + 0.1 * distances
    )
    out = WeightedAdjustedRandIndex().compute(
        y_true, y_pred, scored_intervals=scored_islands(y_true)
    )
    assert out["score"] == expected


def test_masked_f1_ignores_points_outside_the_support_and_never_matches_across_a_gap() -> (
    None
):
    n = 100
    support = [(0, 40), (50, 100)]
    f1 = F1Score(margin=5)
    # 40 and 50 are the gap edges: neither can be judged.
    out = f1.compute(
        [20, 40, 50, 70], [20, 45, 70], n_timepoints=n, scored_intervals=support
    )
    assert out["score"] == 1.0 and out["n_supported"] == 39 + 49
    # A true CP at 38 and a prediction at 52 are within the margin but on
    # either side of the gap.
    support = [(0, 40), (41, 100)]
    out = F1Score(margin=20).compute(
        [38], [42], n_timepoints=n, scored_intervals=support
    )
    assert out["score"] == 0.0
    out = F1Score(margin=20).compute([38], [42], n_timepoints=n)
    assert out["score"] == 1.0


def test_masked_bic_ignores_a_change_point_inside_a_gap() -> None:
    n = 100
    support = [(0, 40), (60, 100)]
    bic = BidirectionalCovering()
    base = bic.compute([20, 80], [20, 80], n_timepoints=n, scored_intervals=support)
    assert base["score"] == 1.0 and base["n_scored"] == 80
    gap = bic.compute([20, 80], [20, 50, 80], n_timepoints=n, scored_intervals=support)
    assert gap["score"] == 1.0
    inside = bic.compute(
        [20, 80], [20, 30, 80], n_timepoints=n, scored_intervals=support
    )
    assert inside["score"] < 1.0
    covering = Covering().compute(
        [20, 80], [20, 50, 80], n_timepoints=n, scored_intervals=support
    )
    assert covering["score"] == 1.0


@pytest.mark.parametrize("make", CPD_METRICS, ids=["F1", "Covering", "BiC"])
def test_change_point_metrics_need_the_length_and_a_supported_position(make) -> None:
    with pytest.raises(ValueError, match="n_timepoints"):
        make().compute([5], [5], scored_intervals=[(0, 10)])
    with pytest.raises(ValueError, match="no change-point position"):
        make().compute([5], [5], n_timepoints=10, scored_intervals=[(2, 3), (5, 6)])


@pytest.mark.parametrize("metric", SD_METRICS, ids=lambda m: m.__name__)
def test_a_reference_without_scored_point_is_refused(metric) -> None:
    y = np.full(6, -1)
    with pytest.raises(ValueError, match="no scored sample"):
        metric().compute(y, np.zeros(6, dtype=int), scored_intervals=scored_islands(y))
