"""Tests of AutoPlait, a Python translation of the authors' C code.

The expected segmentations are those of the C binary (version of June 18,
2014, built with GCC 11 on Linux) on the same series, written as float32
values: the translation draws the same random numbers and returns the same
segments and the same total cost.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("numba")

from tsseg import load_mocap  # noqa: E402
from tsseg.algorithms.autoplait import _plait  # noqa: E402
from tsseg.algorithms.autoplait.detector import (  # noqa: E402
    AutoPlaitDetector,
    _regime_labels,
)
from tsseg.algorithms.utils import extract_cps  # noqa: E402


def _series(name):
    if name == "ABA":
        rng = np.random.default_rng(1)
        return np.concatenate(
            [rng.normal(0, 1, 300), rng.normal(5, 1, 300), rng.normal(0, 1, 300)]
        )
    if name == "ABCA":
        rng = np.random.default_rng(2)
        a = rng.normal(0, 1, (300, 3))
        b = rng.normal(3, 1, (300, 3))
        c = rng.normal(0, 3, (300, 3))
        return np.concatenate([a, b, c, rng.normal(0, 1, (300, 3))])
    if name == "noise":
        return np.random.default_rng(3).normal(0, 1, (900, 2))
    if name == "mocap0":
        return load_mocap(trial=0)[0]
    raise KeyError(name)


# (start, length) of the segments of each regime, in the order of the binary's
# segment.<r> files, and the total cost it prints (rounded to the bit).
BINARY_OUTPUTS = {
    "ABA": ([[(300, 300)], [(0, 300), (600, 300)]], 1046),
    "ABCA": ([[(600, 300)], [(0, 300), (900, 300)], [(300, 300)]], 6002),
    "noise": ([[(0, 900)]], 3933),
    "mocap0": (
        [
            [(3252, 633)],
            [(0, 589), (1149, 835), (2453, 799)],
            [(589, 560)],
            [(1984, 469), (3885, 694)],
        ],
        22685,
    ),
}


def test_rand_reproduces_the_gnu_c_library():
    state = _plait._srand(1)
    values = [int(_plait._rand(state)) for _ in range(5)]
    assert values == [1804289383, 846930886, 1681692777, 1714636915, 1957747793]


@pytest.mark.parametrize("name", sorted(BINARY_OUTPUTS))
def test_segments_match_the_c_binary(name):
    regimes, cost = BINARY_OUTPUTS[name]

    result = _plait.autoplait(_series(name))

    assert [list(zip(s.st, s.ln)) for s in result.regimes] == regimes
    assert round(result.costT) == cost


def test_labels_are_the_regime_of_each_segment():
    detector = AutoPlaitDetector()

    y = detector.fit_predict(_series("ABCA"))

    expected = [1] * 300 + [2] * 300 + [0] * 300 + [1] * 300
    np.testing.assert_array_equal(y, expected)
    assert y.dtype.kind == "i"
    assert detector.n_regimes_ == 3
    assert round(detector.cost_) == 6002
    np.testing.assert_array_equal(extract_cps(y), [300, 600, 900])


def test_more_channels_than_the_binary_reads():
    """The binary counts a row of more than 252 characters as several rows."""
    X = np.random.default_rng(5).normal(0, 1, (300, 76)) * 1e-5

    y = AutoPlaitDetector().fit_predict(X)

    assert y.shape == (300,)
    assert y.dtype.kind == "i"


def test_a_constant_channel_is_ignored():
    X = np.column_stack([_series("ABA"), np.full(900, 3.0)])

    y = AutoPlaitDetector().fit_predict(X)

    np.testing.assert_array_equal(extract_cps(y), [300, 600])


def test_runs_without_n_cps_or_y():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        y = AutoPlaitDetector().fit_predict(_series("ABA"))

    np.testing.assert_array_equal(extract_cps(y), [300, 600])


@pytest.mark.parametrize("n_cps", [0, 2])
def test_n_cps_is_deprecated_and_ignored(n_cps):
    """A harness in guided mode still passes ``n_cps``: warn, run unguided."""
    with pytest.warns(FutureWarning, match="n_cps"):
        y = AutoPlaitDetector(n_cps=n_cps).fit_predict(_series("ABA"))

    np.testing.assert_array_equal(extract_cps(y), [300, 600])


def test_tags_declare_an_unsupervised_detector():
    detector = AutoPlaitDetector()

    assert detector.get_tag("capability:unsupervised") is True
    assert detector.get_tag("capability:semi_supervised") is False
    assert detector.get_tag("python_dependencies") == ["numba"]


@pytest.mark.parametrize(
    "regimes",
    [
        [[(0, 300)], [(300, 299)]],  # last point missing
        [[(0, 300)], [(299, 301)]],  # point 299 twice
        [[(0, 300)], [(300, 301)]],  # past the end
        [[(0, 300)], [(300, 300), (600, 51)]],  # starts at n
        [[(-1, 300)], [(300, 300)]],  # starts before 0
    ],
)
def test_segments_that_do_not_tile_the_series_raise(regimes):
    with pytest.raises(RuntimeError, match="segment"):
        _regime_labels(regimes, 600)
