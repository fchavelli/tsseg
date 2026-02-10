"""Tests for tsseg.metrics — change-point and state-detection metrics."""

import pytest
import numpy as np

from tsseg.metrics import (
    F1Score,
    Covering,
    HausdorffDistance,
    BidirectionalCovering,
    StateMatchingScore,
    AdjustedRandIndex,
    NormalizedMutualInformation,
    AdjustedMutualInformation,
    WeightedAdjustedRandIndex,
    WeightedNormalizedMutualInformation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cp_data():
    """Change-point lists with series_length=50 as last element.

    Convention: the last element of y_true is the series length (boundary)
    and is stripped before matching by F1Score.
    """
    y_true = [10, 20, 30, 50]  # 50 = boundary
    y_pred = [11, 23, 30, 50]  # 50 = boundary
    return y_true, y_pred


@pytest.fixture
def state_data():
    """Imperfect but overlapping state labels."""
    length = 50
    y_true = np.zeros(length, dtype=int)
    y_true[10:25] = 1
    y_true[40:50] = 2

    y_pred = np.zeros(length, dtype=int)
    y_pred[12:28] = 1
    y_pred[38:50] = 2
    return y_true, y_pred


@pytest.fixture
def state_data_perfect():
    """Identical true and predicted state labels."""
    length = 50
    y_true = np.zeros(length, dtype=int)
    y_true[10:25] = 1
    y_true[40:50] = 2
    return y_true, y_true.copy()


# ---------------------------------------------------------------------------
# Change-Point Detection metrics
# ---------------------------------------------------------------------------

class TestF1Score:
    """F1Score returns keys: score, precision, recall."""

    def test_within_margin(self, cp_data):
        y_true, y_pred = cp_data
        result = F1Score(margin=5).compute(y_true, y_pred)
        assert result["score"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_perfect(self, cp_data):
        y_true, _ = cp_data
        result = F1Score(margin=5).compute(y_true, y_true)
        assert result["score"] == 1.0

    def test_empty_lists(self):
        f1 = F1Score()
        assert f1.compute([], [])["score"] == 1.0
        assert f1.compute([1], [])["score"] == 0.0
        assert f1.compute([], [1])["score"] == 0.0


class TestCovering:
    """Covering returns key: score."""

    def test_partial_overlap(self):
        y_true = [0, 10, 20, 30]
        y_pred = [0, 15, 30]
        result = Covering().compute(y_true, y_pred)
        assert result["score"] == pytest.approx(0.5277777777777778)

    def test_perfect(self):
        y_true = [0, 10, 20, 30]
        result = Covering().compute(y_true, y_true)
        assert result["score"] == 1.0


class TestBidirectionalCovering:
    """BidirectionalCovering returns: score, ground_truth_covering, prediction_covering."""

    def test_partial(self):
        y_true = [0, 10, 20, 30]
        y_pred = [0, 15, 30]
        result = BidirectionalCovering().compute(y_true, y_pred)

        assert result["ground_truth_covering"] == pytest.approx(0.5277777777777778)
        assert result["prediction_covering"] == pytest.approx(2 / 3)
        assert result["score"] == pytest.approx(76 / 129)

    def test_arithmetic_aggregation(self):
        y_true = [0, 10, 20, 30]
        y_pred = [0, 15, 30]
        result = BidirectionalCovering(aggregation="arithmetic").compute(y_true, y_pred)

        expected = 0.5 * (result["ground_truth_covering"] + result["prediction_covering"])
        assert result["score"] == pytest.approx(expected)


class TestHausdorffDistance:
    """HausdorffDistance returns: score, hausdorff_distance."""

    def test_known_distance(self):
        y_true = [10, 20, 30, 50]
        y_pred = [11, 23, 30, 45]
        result = HausdorffDistance().compute(y_true, y_pred)
        assert result["hausdorff_distance"] == 5
        assert result["score"] == 5


# ---------------------------------------------------------------------------
# State Detection metrics
# ---------------------------------------------------------------------------

class TestAdjustedRandIndex:
    """ARI returns key: score."""

    def test_imperfect(self, state_data):
        result = AdjustedRandIndex().compute(*state_data)
        assert "score" in result
        assert result["score"] == pytest.approx(0.587, abs=1e-3)

    def test_perfect(self, state_data_perfect):
        result = AdjustedRandIndex().compute(*state_data_perfect)
        assert result["score"] == 1.0


class TestNormalizedMutualInformation:
    def test_imperfect(self, state_data):
        result = NormalizedMutualInformation().compute(*state_data)
        assert "score" in result
        assert result["score"] == pytest.approx(0.603, abs=1e-3)


class TestAdjustedMutualInformation:
    def test_imperfect(self, state_data):
        result = AdjustedMutualInformation().compute(*state_data)
        assert "score" in result
        assert result["score"] == pytest.approx(0.586, abs=1e-3)


class TestStateMatchingScore:
    """SMS returns key: score."""

    def test_imperfect(self, state_data):
        result = StateMatchingScore().compute(*state_data)
        assert "score" in result
        assert result["score"] > 0.8

    def test_perfect(self, state_data_perfect):
        result = StateMatchingScore().compute(*state_data_perfect)
        assert result["score"] == 1.0


class TestWeightedAdjustedRandIndex:
    def test_runs(self, state_data):
        result = WeightedAdjustedRandIndex().compute(*state_data)
        assert "score" in result
        assert isinstance(result["score"], float)


class TestWeightedNormalizedMutualInformation:
    def test_runs(self, state_data):
        result = WeightedNormalizedMutualInformation().compute(*state_data)
        assert "score" in result
        assert isinstance(result["score"], float)
