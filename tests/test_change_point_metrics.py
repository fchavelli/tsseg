"""Edge cases of the change-point metrics in ``tsseg.metrics.change_point_detection``.

The main behaviour is tested in ``test_metrics.py`` and
``test_metrics_series_length.py``; this file covers input validation,
margins, matching and the degenerate cases without change points.
"""

import warnings

import numpy as np
import pytest

from tsseg.metrics import Covering, F1Score, HausdorffDistance

legacy_series_length = pytest.mark.filterwarnings(
    "ignore:.*series length is inferred:UserWarning"
)


@pytest.mark.parametrize("metric", [F1Score(), Covering()])
@pytest.mark.parametrize("n_timepoints", [0, -5])
def test_n_timepoints_must_be_positive(metric, n_timepoints):
    with pytest.raises(ValueError, match="n_timepoints must be >= 1"):
        metric.compute([], [], n_timepoints=n_timepoints)


@legacy_series_length
class TestF1Margin:
    @pytest.mark.parametrize("margin", [-0.1, 1.5])
    def test_float_margin_outside_unit_interval(self, margin):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            F1Score(margin=margin).compute([20, 50], [20])

    def test_negative_integer_margin(self):
        with pytest.raises(ValueError, match="non-negative"):
            F1Score(margin=-1).compute([20, 50], [20])

    def test_float_margin_is_a_fraction_of_the_series_length(self):
        # 0.05 * 100 = 5 samples: 24 matches 20, 26 does not.
        assert F1Score(margin=0.05).compute([20, 100], [24])["score"] == 1.0
        assert F1Score(margin=0.05).compute([20, 100], [26])["score"] == 0.0

    def test_zero_margin_needs_an_exact_hit(self):
        assert F1Score(margin=0).compute([20, 50], [20])["score"] == 1.0
        assert F1Score(margin=0).compute([20, 50], [21])["score"] == 0.0
        assert F1Score(margin=1).compute([20, 50], [21])["score"] == 1.0

    def test_margin_rounded_to_zero_warns_when_matching(self):
        # int(0.01 * 50) == 0: the tolerance has no effect.
        with pytest.warns(UserWarning, match="margin equals 0 samples"):
            result = F1Score(margin=0.01).compute([20, 50], [21])
        assert result["score"] == 0.0

    @pytest.mark.parametrize(
        "y_true, y_pred", [([50], []), ([50], [50]), ([20, 50], [])]
    )
    def test_margin_rounded_to_zero_silent_without_matching(self, y_true, y_pred):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*margin equals 0")
            F1Score(margin=0.01).compute(y_true, y_pred)


@legacy_series_length
class TestF1Matching:
    def test_a_prediction_matches_one_true_change_point_only(self):
        # 21 lies within 5 samples of 20 and 22 but counts once.
        result = F1Score(margin=5).compute([20, 22, 100], [21])
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert result["score"] == pytest.approx(2 / 3)

    def test_greedy_matching_of_the_closest_prediction(self):
        # Greedy, as in the reference code (van den Burg & Williams, 2020):
        # each true change point (here in increasing order) takes its closest
        # free prediction. 20 takes 21 (not 17), which leaves nothing within 3
        # of 24; a maximum matching (20-17, 24-21) would find 2 true positives.
        result = F1Score(margin=3).compute([20, 24, 100], [17, 21])
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5

    def test_duplicate_prediction_is_a_false_positive(self):
        # A repeated change point is matched once; its copy still counts in
        # the precision.
        result = F1Score(margin=2).compute([20, 100], [20, 20])
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0

    def test_numpy_inputs(self):
        y_true, y_pred = np.array([20, 60, 100]), np.array([21, 80])
        expected = F1Score(margin=2).compute(y_true.tolist(), y_pred.tolist())
        assert F1Score(margin=2).compute(y_true, y_pred) == expected
        assert expected["score"] == 0.5


@legacy_series_length
class TestNoChangePoint:
    """Without any change point, F1Score and Covering agree on a perfect score."""

    @pytest.mark.parametrize("y_pred", [[], [50], [0, 50]])
    def test_series_length_read_from_y_true(self, y_pred):
        assert F1Score().compute([50], y_pred)["score"] == 1.0
        assert Covering().compute([50], y_pred)["score"] == 1.0

    def test_spurious_prediction(self):
        assert F1Score().compute([50], [25])["score"] == 0.0
        assert Covering().compute([50], [25])["score"] == pytest.approx(0.5)


class TestHausdorffDistance:
    @pytest.mark.parametrize("y_true, y_pred", [([], [5]), ([5], []), ([], [])])
    def test_empty_side_is_infinitely_far(self, y_true, y_pred):
        result = HausdorffDistance().compute(y_true, y_pred)
        assert result == {"score": np.inf, "hausdorff_distance": np.inf}

    def test_largest_of_both_directed_distances(self):
        # true -> pred: 10->12 (2), 50->40 (10); pred -> true: 12 (2), 40 (10),
        # 90->50 (40).
        result = HausdorffDistance().compute([10, 50], [12, 40, 90])
        assert result["score"] == 40
