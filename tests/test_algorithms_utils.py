"""Unit tests of the helpers in ``tsseg.algorithms.utils``.

``multivariate_l2_norm`` (BOCD, BEAST, Prophet, FLUSS, ChangeFinder),
``aggregate_change_points`` (public flat helper, kept for direct callers),
``_fuse_channel_change_points`` (ensembling of BOCD, BEAST, FLUSS and
ChangeFinder), ``consensus_change_points`` (FLUSS, ChangeFinder),
``create_state_labels`` and ``extract_cps`` (ground truth of the documentation
and of the test suite).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms.utils import (
    _fuse_channel_change_points,
    aggregate_change_points,
    consensus_change_points,
    create_state_labels,
    extract_cps,
    multivariate_l2_norm,
)
from tsseg.metrics.change_point_detection import labels_to_change_points


class TestMultivariateL2Norm:
    def test_norm_of_each_time_point(self):
        signal = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(
            multivariate_l2_norm(signal), [5.0, 0.0, 1.0, np.sqrt(2.0)]
        )

    def test_univariate_signal_is_returned_unchanged(self):
        signal = np.array([1.0, -2.0, 3.0])
        assert multivariate_l2_norm(signal) is signal

    def test_single_channel_gives_absolute_values(self):
        signal = np.array([[1.0], [-2.0], [3.0]])
        np.testing.assert_allclose(multivariate_l2_norm(signal), [1.0, 2.0, 3.0])


class TestAggregateChangePoints:
    def test_no_change_point(self):
        result = aggregate_change_points([], n_cp=3)
        assert result.shape == (0,)
        assert result.dtype.kind == "i"

    def test_exact_mode_keeps_the_most_frequent_points(self):
        # 50 seen 3 times, 80 twice, 20 and 95 once.
        result = aggregate_change_points([80, 50, 20, 50, 95, 80, 50], n_cp=2)
        np.testing.assert_array_equal(result, [50, 80])
        assert result.dtype.kind == "i"

    def test_exact_mode_breaks_ties_by_position(self):
        # All seen once: the earliest ones are kept.
        result = aggregate_change_points([70, 10, 40], n_cp=2)
        np.testing.assert_array_equal(result, [10, 40])

    def test_n_cp_larger_than_the_candidates(self):
        result = aggregate_change_points([30, 10, 30], n_cp=5)
        np.testing.assert_array_equal(result, [10, 30])

    def test_zero_change_points_requested(self):
        assert aggregate_change_points([10, 20], n_cp=0).size == 0
        assert aggregate_change_points([10, 20], n_cp=0, tolerance=5).size == 0

    def test_tolerance_groups_nearby_points(self):
        # Clusters {48, 50, 53} (3 points), {200, 203} (2), {400} (1).
        cps = [50, 203, 400, 48, 200, 53]
        result = aggregate_change_points(cps, n_cp=2, tolerance=5)
        np.testing.assert_array_equal(result, [50, 201])

    def test_cluster_representative_is_the_truncated_median(self):
        # median(10, 13) = 11.5, truncated to 11.
        result = aggregate_change_points([10, 13], n_cp=1, tolerance=3)
        np.testing.assert_array_equal(result, [11])

    def test_clusters_are_chained(self):
        # Successive gaps of 4 <= tolerance: one cluster spanning 12 samples.
        result = aggregate_change_points([10, 14, 18, 22], n_cp=4, tolerance=4)
        np.testing.assert_array_equal(result, [16])

    def test_cluster_ties_are_broken_by_position(self):
        result = aggregate_change_points([300, 100, 200], n_cp=2, tolerance=5)
        np.testing.assert_array_equal(result, [100, 200])

    def test_fractional_tolerance_is_relative_to_the_signal_length(self):
        # 0.01 * 1000 = 10 samples.
        result = aggregate_change_points(
            [100, 108, 300], n_cp=1, tolerance=0.01, signal_len=1000
        )
        np.testing.assert_array_equal(result, [104])

    def test_fractional_tolerance_needs_the_signal_length(self):
        with pytest.raises(ValueError, match="signal_len"):
            aggregate_change_points([10, 20], n_cp=1, tolerance=0.1)

    def test_fractional_tolerance_below_one_sample_is_exact(self):
        # 0.001 * 100 = 0.1, truncated to 0: exact matching.
        result = aggregate_change_points(
            [10, 11, 11], n_cp=1, tolerance=0.001, signal_len=100
        )
        np.testing.assert_array_equal(result, [11])

    def test_float_tolerance_of_at_least_one_is_in_samples(self):
        as_float = aggregate_change_points([10, 12, 50], n_cp=1, tolerance=2.0)
        as_int = aggregate_change_points([10, 12, 50], n_cp=1, tolerance=2)
        np.testing.assert_array_equal(as_float, as_int)
        np.testing.assert_array_equal(as_float, [11])


class TestConsensusChangePoints:
    def test_no_change_point(self):
        result = consensus_change_points([[], []], tolerance=5, consensus=0.5)
        assert result.shape == (0,)
        assert result.dtype == np.int64

    def test_no_channel(self):
        assert consensus_change_points([], tolerance=5, consensus=0.5).size == 0

    def test_points_supported_by_enough_channels(self):
        # 3 channels, consensus 0.5: at least ceil(1.5) = 2 channels.
        per_channel = [[100, 500], [102, 700], [98, 503]]
        result = consensus_change_points(per_channel, tolerance=5, consensus=0.5)
        # 500 and 503 are both supported by 2 channels with the same strength:
        # the earlier observed index is kept, not their median 501.
        np.testing.assert_array_equal(result, [100, 500])
        assert result.dtype == np.int64

    def test_one_channel_counts_once_per_cluster(self):
        # Two points of channel 0 in one cluster: a single supporting channel.
        per_channel = [[100, 103], [400]]
        result = consensus_change_points(per_channel, tolerance=5, consensus=1.0)
        assert result.size == 0

    def test_full_consensus(self):
        per_channel = [[100, 300], [101, 600], [99, 300]]
        result = consensus_change_points(per_channel, tolerance=3, consensus=1.0)
        np.testing.assert_array_equal(result, [100])

    def test_zero_consensus_keeps_every_cluster(self):
        per_channel = [[100], [300], [500]]
        result = consensus_change_points(per_channel, tolerance=3, consensus=0.0)
        np.testing.assert_array_equal(result, [100, 300, 500])

    def test_rounding_of_the_channel_count(self):
        # 28 % of 25 channels is 7 channels, although 0.28 * 25 evaluates to
        # 7.000000000000001 in floating point.
        per_channel = [[100]] * 7 + [[]] * 18
        result = consensus_change_points(per_channel, tolerance=0, consensus=0.28)
        np.testing.assert_array_equal(result, [100])

    def test_zero_tolerance_groups_identical_points_only(self):
        per_channel = [[100], [101]]
        result = consensus_change_points(per_channel, tolerance=0, consensus=1.0)
        assert result.size == 0

    def test_candidate_within_tolerance_of_every_channel(self):
        # 14 is within 4 samples of the change points of the three channels.
        per_channel = [[10], [14], [18]]
        result = consensus_change_points(per_channel, tolerance=4, consensus=1.0)
        np.testing.assert_array_equal(result, [14])

    def test_points_are_not_chained(self):
        # Successive gaps of 4 used to chain 10..22 into one cluster of four
        # channels; no observed point is within 4 samples of all of them.
        per_channel = [[10], [14], [18], [22]]
        result = consensus_change_points(per_channel, tolerance=4, consensus=1.0)
        assert result.size == 0
        result = consensus_change_points(per_channel, tolerance=4, consensus=0.0)
        np.testing.assert_array_equal(result, [14, 22])

    def test_accepts_arrays(self):
        per_channel = [np.array([100, 500]), np.array([101])]
        result = consensus_change_points(per_channel, tolerance=2, consensus=1.0)
        np.testing.assert_array_equal(result, [100])


class TestFuseChannelChangePoints:
    """Per-channel fusion used by the ensembling strategies."""

    def test_no_channel_and_no_change_point(self):
        for per_channel in ([], [[], []]):
            result = _fuse_channel_change_points(per_channel, 0, n_cp=3)
            assert result.shape == (0,)
            assert result.dtype == np.int64

    def test_zero_cap(self):
        assert _fuse_channel_change_points([[10], [20]], 0, n_cp=0).size == 0

    def test_zero_tolerance_is_an_exact_vote(self):
        # No exact coincidence: ties are broken by position, as before.
        per_channel = [[100, 1000, 3000], [250, 1001, 2999], [400, 998, 3002]]
        result = _fuse_channel_change_points(per_channel, 0, n_cp=2)
        np.testing.assert_array_equal(result, [100, 250])

    def test_zero_tolerance_matches_the_flat_helper(self):
        per_channel = [[80, 50], [20, 50, 95], [80, 50]]
        flat = [cp for cps in per_channel for cp in cps]
        for n_cp in range(5):
            np.testing.assert_array_equal(
                _fuse_channel_change_points(per_channel, 0, n_cp=n_cp),
                aggregate_change_points(flat, n_cp=n_cp),
            )

    def test_distinct_channels_beat_points_of_one_channel(self):
        # Three points of channel 0 used to outvote two channels at 100.
        per_channel = [[10, 11, 12], [100], [100]]
        result = _fuse_channel_change_points(per_channel, 4, n_cp=1)
        np.testing.assert_array_equal(result, [100])

    def test_duplicates_do_not_add_votes(self):
        per_channel = [[10, 10, 10], [20], [20]]
        result = _fuse_channel_change_points(per_channel, 0, n_cp=1)
        np.testing.assert_array_equal(result, [20])

    def test_finite_tolerance_does_not_chain(self):
        per_channel = [[10], [14], [18], [22]]
        result = _fuse_channel_change_points(per_channel, 4, n_cp=4)
        np.testing.assert_array_equal(result, [14, 22])

    def test_the_elected_index_is_an_observed_one(self):
        # The median of 10, 15, 20, 20 was 17, detected by no channel.
        per_channel = [[10], [15], [20], [20]]
        result = _fuse_channel_change_points(per_channel, 5, n_cp=1)
        np.testing.assert_array_equal(result, [15])

    def test_support_comes_before_strength(self):
        # 100 is supported by 3 channels (at distances 0, 5, 5), 200 by 2 at 0.
        per_channel = [[100], [95], [105], [200], [200]]
        result = _fuse_channel_change_points(per_channel, 5, n_cp=1)
        np.testing.assert_array_equal(result, [100])

    def test_strength_breaks_ties_of_support(self):
        # Both pairs have 2 channels; 200 and 201 are closer than 100 and 105.
        per_channel = [[100], [105], [200], [201]]
        result = _fuse_channel_change_points(per_channel, 5, n_cp=1)
        np.testing.assert_array_equal(result, [200])

    def test_exclusion_radius_is_closed(self):
        # 15 is exactly 5 samples after 10 and is excluded; 21 is 6 after.
        result = _fuse_channel_change_points([[10, 15, 21]], 5, n_cp=3)
        np.testing.assert_array_equal(result, [10, 21])

    def test_no_change_point_is_created_to_fill_the_cap(self):
        result = _fuse_channel_change_points([[100]], 10, n_cp=3)
        np.testing.assert_array_equal(result, [100])

    def test_greedy_selection_may_return_fewer_than_feasible(self):
        # 15 has the best support and excludes 10 and 20, although 10 and 20
        # together would satisfy the cap of 2.
        result = _fuse_channel_change_points([[10], [15], [20]], 5, n_cp=2)
        np.testing.assert_array_equal(result, [15])

    def test_provenance_changes_the_result_of_the_same_flat_input(self):
        one_channel = [[10, 12, 14], [100], []]
        three_channels = [[10, 100], [12], [14]]
        np.testing.assert_array_equal(
            _fuse_channel_change_points(one_channel, 4, n_cp=1), [10]
        )
        np.testing.assert_array_equal(
            _fuse_channel_change_points(three_channels, 4, n_cp=1), [12]
        )

    def test_consensus_counts_the_empty_channels(self):
        result = _fuse_channel_change_points([[100], [], []], 5, consensus=0.5)
        assert result.size == 0
        result = _fuse_channel_change_points([[100], [], []], 5, consensus=0.3)
        np.testing.assert_array_equal(result, [100])

    def test_unobserved_midpoints_are_not_candidates(self):
        result = _fuse_channel_change_points([[10], [20]], 5, consensus=1.0)
        assert result.size == 0

    def test_no_cap_keeps_every_elected_candidate(self):
        result = _fuse_channel_change_points([[100], [300], [500]], 3)
        np.testing.assert_array_equal(result, [100, 300, 500])

    @pytest.mark.parametrize(
        ("per_channel", "tolerance", "n_cp", "consensus"),
        [
            ([[10, 11, 12], [100], [100]], 4, 1, 0.0),
            ([[10], [14], [18], [22]], 4, 4, 0.0),
            ([[100], [95], [105], [200], [200]], 5, None, 0.4),
            ([[10, 15, 21], [16], [3, 30]], 5, 3, 0.0),
        ],
    )
    def test_invariant_to_order_and_duplicates(
        self, per_channel, tolerance, n_cp, consensus
    ):
        expected = _fuse_channel_change_points(
            per_channel, tolerance, n_cp=n_cp, consensus=consensus
        )
        reordered = [list(reversed(cps)) for cps in reversed(per_channel)]
        duplicated = [cps + cps for cps in per_channel]
        for variant in (reordered, duplicated):
            np.testing.assert_array_equal(
                _fuse_channel_change_points(
                    variant, tolerance, n_cp=n_cp, consensus=consensus
                ),
                expected,
            )

    def test_negative_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="tolerance"):
            _fuse_channel_change_points([[10]], -1)


class TestCreateStateLabels:
    def test_one_label_per_segment(self):
        labels = create_state_labels([0, 3, 7, 10], 10)
        np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        assert labels.dtype.kind == "i"

    def test_boundaries_are_optional(self):
        np.testing.assert_array_equal(
            create_state_labels([3, 7], 10), create_state_labels([0, 3, 7, 10], 10)
        )

    def test_order_and_duplicates_do_not_matter(self):
        np.testing.assert_array_equal(
            create_state_labels([7, 3, 7, 0], 10), create_state_labels([3, 7], 10)
        )

    def test_no_change_point(self):
        np.testing.assert_array_equal(create_state_labels([], 5), np.zeros(5))

    def test_points_beyond_the_series_are_capped(self):
        np.testing.assert_array_equal(
            create_state_labels([0, 6, 15], 8), [0, 0, 0, 0, 0, 0, 1, 1]
        )

    def test_labels_are_segment_indices_not_recurring_states(self):
        # Three segments, three labels: no state is ever reused.
        labels = create_state_labels([4, 8], 12)
        assert np.unique(labels).tolist() == [0, 1, 2]


class TestExtractCps:
    def test_first_index_of_each_new_label(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 2, 2])
        np.testing.assert_array_equal(extract_cps(labels), [2, 5, 6])

    def test_constant_sequence(self):
        assert extract_cps(np.zeros(10, dtype=int)).size == 0

    @pytest.mark.parametrize("n", [0, 1])
    def test_sequences_too_short_for_a_change(self, n):
        assert extract_cps(np.zeros(n, dtype=int)).size == 0

    def test_float_labels(self):
        np.testing.assert_array_equal(extract_cps(np.array([0.0, 0.0, 1.5])), [2])

    def test_agrees_with_labels_to_change_points(self):
        # labels_to_change_points adds the boundaries 0 and n.
        labels = np.array([3, 3, 1, 1, 1, 3, 3, 0])
        np.testing.assert_array_equal(
            labels_to_change_points(labels),
            [0, *extract_cps(labels).tolist(), labels.size],
        )

    @pytest.mark.parametrize("seed", range(5))
    def test_inverse_of_create_state_labels(self, seed):
        rng = np.random.default_rng(seed)
        n = 200
        cps = np.sort(
            rng.choice(np.arange(1, n), size=rng.integers(0, 10), replace=False)
        )
        labels = create_state_labels(cps.tolist(), n)
        np.testing.assert_array_equal(extract_cps(labels), cps)
