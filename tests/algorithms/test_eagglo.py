"""Penalties and initial blocks of EAggloDetector."""

import numpy as np
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


@pytest.mark.xfail(
    strict=True,
    reason="predict() reuses the fit() result for any series of the same length "
    "(the index is compared, not the values)",
)
def test_predict_on_another_series_of_the_same_length():
    rng = np.random.default_rng(0)
    x1 = np.r_[rng.normal(0, 1, 60), rng.normal(6, 1, 60)]
    x2 = np.r_[rng.normal(0, 1, 30), rng.normal(6, 1, 90)]
    member = np.repeat(np.arange(12), 10)
    detector = EAggloDetector(member=member).fit(x1)
    expected = EAggloDetector(member=member).fit_predict(x2)
    assert expected.tolist() == [30]
    assert detector.predict(x2).tolist() == expected.tolist()
