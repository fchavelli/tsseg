"""Penalties and initial blocks of EAggloDetector."""

import numpy as np
import pytest

pytest.importorskip("numba")

from tsseg.algorithms import EAggloDetector  # noqa: E402


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
