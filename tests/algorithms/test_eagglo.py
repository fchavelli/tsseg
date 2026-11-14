"""Penalties, initial blocks and predict() of EAggloDetector."""

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("numba")

from scipy.spatial.distance import cdist  # noqa: E402

from tsseg.algorithms import EAggloDetector  # noqa: E402
from tsseg.algorithms.eagglo.detector import (  # noqa: E402
    euclidean,
    euclidean_matrix_to_matrix,
    get_distance_matrix,
    get_distance_single,
    len_penalty,
    mean_diff_penalty,
)


def _three_segments(rng=0):
    rng = np.random.default_rng(rng)
    return np.concatenate(
        [rng.normal(0, 1, 100), rng.normal(3, 1, 100), rng.normal(0, 3, 100)]
    )


@pytest.mark.parametrize("penalty", ["len_penalty", "mean_diff_penalty"])
def test_named_penalties_run(penalty):
    x = _three_segments()
    pred = np.asarray(EAggloDetector(penalty=penalty).fit_predict(x))
    assert np.all((pred > 0) & (pred < len(x)))


def test_callable_penalty_receives_the_change_points():
    x = _three_segments()
    seen = []

    def penalty(cps):
        seen.append(np.asarray(cps))
        return 0.0

    EAggloDetector(penalty=penalty).fit(x)
    assert len(seen) == len(x)
    assert all(np.all(np.diff(np.sort(cps)) > 0) for cps in seen)


def test_initial_blocks():
    # from single observations the goodness of fit keeps a very fine
    # segmentation; from blocks of 10 it recovers the two change points
    x = _three_segments()
    assert len(EAggloDetector().fit_predict(x)) > 50
    member = np.repeat(np.arange(30), 10)
    assert list(EAggloDetector(member=member).fit_predict(x)) == [100, 200]


def test_penalty_functions():
    assert len_penalty([0, 10, 30]) == -3
    assert mean_diff_penalty([30, 0, 10]) == pytest.approx(15.0)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_distance_helpers_match_scipy(alpha):
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(4, 3)), rng.normal(size=(5, 3))
    expected = cdist(a, b)
    np.testing.assert_allclose(euclidean_matrix_to_matrix(a, b), expected)
    assert euclidean(a[0], b[0]) == pytest.approx(expected[0, 0])
    assert get_distance_single(a[1], b[2], alpha) == pytest.approx(
        expected[1, 2] ** alpha
    )
    assert get_distance_matrix(a, b, alpha) == pytest.approx((expected**alpha).mean())


def test_unsorted_initial_blocks_are_rejected():
    member = np.repeat([1, 0, 2], 100)
    with pytest.raises(ValueError, match="sorted"):
        EAggloDetector(member=member).fit(_three_segments())


def test_predict_on_a_longer_series_refits_with_a_warning():
    x = _three_segments()
    detector = EAggloDetector().fit(x[:200])
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        pred = detector.predict(x)
    assert np.all((pred > 0) & (pred < len(x)))
    assert np.any(pred >= 200)


def _two_series_of_the_same_length():
    rng = np.random.default_rng(0)
    x1 = np.r_[rng.normal(0, 1, 60), rng.normal(6, 1, 60)]
    x2 = np.r_[rng.normal(0, 1, 30), rng.normal(6, 1, 90)]
    return x1, x2


def test_predict_on_another_series_of_the_same_length():
    # the index of both series is 0..119: predict() used to compare it and
    # return the change point of x1 (60)
    x1, x2 = _two_series_of_the_same_length()
    member = np.repeat(np.arange(12), 10)
    detector = EAggloDetector(member=member).fit(x1)
    expected = EAggloDetector(member=member).fit_predict(x2)
    assert expected.tolist() == [30]
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        assert detector.predict(x2).tolist() == expected.tolist()
    # the fitted attributes still describe x1
    assert detector.predict(x1).tolist() == [60]


def test_predict_on_a_shorter_series_segments_it_alone():
    # predict() used to complete the shorter series with the end of the series
    # given to fit, segment the result and drop the change points past its end
    rng = np.random.default_rng(3)
    x_fit = np.r_[rng.normal(0, 1, 100), rng.normal(3, 1, 100)]
    x_short = np.r_[rng.normal(0, 1, 50), rng.normal(3, 1, 50)]
    expected = EAggloDetector(penalty="len_penalty").fit_predict(x_short)
    assert expected.tolist() == [50, 98]
    detector = EAggloDetector(penalty="len_penalty").fit(x_fit)
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        assert detector.predict(x_short).tolist() == expected.tolist()


def test_predict_after_the_fit_series_is_modified_in_place():
    x1, x2 = _two_series_of_the_same_length()
    x = x1.copy()
    detector = EAggloDetector(member=np.repeat(np.arange(12), 10)).fit(x)
    x[:] = x2
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        assert detector.predict(x).tolist() == [30]


def test_predict_on_the_fit_series_reuses_the_fit():
    x1, _ = _two_series_of_the_same_length()
    detector = EAggloDetector(member=np.repeat(np.arange(12), 10)).fit(x1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # equal values in a new array are the series given to fit
        assert detector.predict(x1.copy()).tolist() == [60]
        assert detector.fit_predict(x1).tolist() == [60]


@pytest.mark.parametrize("n_timepoints", [60, 240])
def test_predict_with_initial_blocks_on_a_series_of_another_length(n_timepoints):
    # member labels the time points of the series given to fit: it cannot
    # segment a series of another length (a shorter one used to be completed
    # with the end of the fitted series)
    x1, _ = _two_series_of_the_same_length()
    detector = EAggloDetector(member=np.repeat(np.arange(12), 10)).fit(x1)
    with pytest.raises(ValueError, match="member has 120 labels"):
        detector.predict(np.resize(x1, n_timepoints))


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("block_sizes", [None, [4, 4, 4], [2, 3, 7]])
def test_initial_distances_match_direct_energy_calculation(alpha, block_sizes):
    """The compiled initialisation retains ecp's unscaled V-distances."""
    x = np.random.default_rng(3).normal(size=(12, 3))
    sizes = np.ones(len(x), dtype=int) if block_sizes is None else block_sizes
    member = None if block_sizes is None else np.repeat([2, 5, 9], sizes)
    detector = EAggloDetector(member=member, alpha=alpha)
    detector._initialize_params(pd.DataFrame(x))

    blocks = np.split(x, np.cumsum(sizes)[:-1])
    within = [(cdist(b, b) ** alpha).mean() for b in blocks]
    expected = np.array(
        [
            [
                2 * (cdist(a, b) ** alpha).mean() - within[i] - within[j]
                for j, b in enumerate(blocks)
            ]
            for i, a in enumerate(blocks)
        ]
    )
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(
        detector.distances[: len(blocks), : len(blocks)], expected, atol=1e-14
    )
    np.testing.assert_array_equal(detector.progression[0], np.r_[0, np.cumsum(sizes)])


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
def test_merges_maximise_the_direct_circular_energy_statistic(alpha):
    """Compare the merge scan with a fresh two-sample calculation at every step.

    For singleton initial clusters, ecp's stored distance is twice the scaled
    V-statistic Q; the circular goodness of fit counts each edge twice again.
    The last (two-to-one) merge is excluded: ecp's recurrence counts overlapping
    neighbours at that step and does not equal the empty sum (zero).
    """
    from tsseg.algorithms.eagglo._kernels import _update_distances

    x = np.random.default_rng(8).normal(size=(9, 2))
    detector = EAggloDetector(alpha=alpha)
    detector._initialize_params(pd.DataFrame(x))
    clusters = {i: x[i : i + 1] for i in range(len(x))}
    order = list(clusters)

    def goodness(partition):
        total = 0.0
        for a, b in zip(partition, partition[1:] + partition[:1], strict=True):
            energy = (
                2 * (cdist(a, b) ** alpha).mean()
                - (cdist(a, a) ** alpha).mean()
                - (cdist(b, b) ** alpha).mean()
            )
            total += 4 * len(a) * len(b) / (len(a) + len(b)) * energy
        return total

    for K in range(len(x) - 1, 2 * len(x) - 3):
        candidates = []
        for i in sorted(clusters):
            position = order.index(i)
            rotated = order[position:] + order[:position]
            j = rotated[1]
            merged = np.concatenate([clusters[i], clusters[j]])
            partition = [merged] + [clusters[k] for k in rotated[2:]]
            candidates.append((goodness(partition), i, j))
        expected_fit, expected_i, expected_j = max(candidates)
        i, j = detector._find_closest(K)
        assert (i, j) == (expected_i, expected_j)
        assert detector.gof_[-1] == pytest.approx(expected_fit)
        _update_distances(
            i,
            j,
            K,
            detector.merged_,
            detector.n_cluster,
            detector.distances,
            detector.left,
            detector.right,
            detector.open,
            detector.sizes,
            detector.progression,
            detector.lm,
        )
        position = order.index(i)
        rotated = order[position:] + order[:position]
        clusters[K + 1] = np.concatenate([clusters.pop(i), clusters.pop(j)])
        order = [K + 1, *rotated[2:]]


def test_equal_merge_scores_keep_the_first_open_cluster():
    """The compiled scan uses a strict comparison, as the Python scan did."""
    detector = EAggloDetector().fit(np.zeros((5, 2)))
    np.testing.assert_array_equal(detector.merged_[0], [0, -1])
    assert detector.gof_.tolist() == [0] * 5
    assert detector.predict(np.zeros((5, 2))).tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("length", [5, 15])
def test_initial_membership_length_is_checked_before_the_compiled_kernel(length):
    with pytest.raises(ValueError, match="Length mismatch"):
        EAggloDetector(member=np.arange(length)).fit(np.zeros((10, 2)))
