"""Tests of SeededBinSegDetector and of its interval systems and selections."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tsseg.algorithms import SeededBinSegDetector
from tsseg.algorithms.ruptures.detection.binseg import Binseg
from tsseg.algorithms.seeded_binseg._core import (
    greedy_path,
    narrowest_over_threshold,
    narrowest_path,
    seeded_intervals,
    ssic,
    wild_intervals,
)


def _as_set(bounds):
    return {(int(s), int(e)) for s, e in bounds}


def _steps(levels, length, sigma, seed=0):
    rng = np.random.default_rng(seed)
    x = np.repeat(np.asarray(levels, dtype=float), length)
    return x + sigma * rng.standard_normal(x.size)


# --- seeded intervals (Kovács et al., 2023, Definition 1) -------------------


def test_seeded_intervals_dyadic_layers():
    # a = 1/2: each interval of a layer split into left, middle and right halves
    expected = {(0, 8), (0, 4), (2, 6), (4, 8)} | {(i, i + 2) for i in range(7)}
    assert _as_set(seeded_intervals(8, 0.5, 2)) == expected


def test_seeded_intervals_layer_counts_ignore_rounding_noise():
    # (1/a)^(k-1) = sqrt(2)^2 is 2.0000000000000004 in floating point: the
    # third layer still has 2 * 2 - 1 = 3 intervals of length n / 2
    bounds = seeded_intervals(1024, 2**-0.5, 2)
    lengths = bounds[:, 1] - bounds[:, 0]
    assert np.sum(lengths == 512) == 3
    assert np.sum(lengths == 1024) == 1


def test_seeded_intervals_contain_the_dyadic_system():
    # Fig. 1 of the paper: the odd layers of a = 1/sqrt(2) are those of a = 1/2
    dyadic = _as_set(seeded_intervals(1024, 0.5, 2))
    assert dyadic <= _as_set(seeded_intervals(1024, 2**-0.5, 2))


@pytest.mark.parametrize("n", [140, 497, 2048])
@pytest.mark.parametrize("decay", [0.5, 2**-0.5, 2**-0.125])
def test_seeded_intervals_near_linear_total_length(n, decay):
    bounds = seeded_intervals(n, decay, 2)
    assert bounds[:, 0].min() == 0 and bounds[:, 1].max() == n
    assert np.all(bounds[:, 1] - bounds[:, 0] >= 2)
    n_layers = math.ceil(math.log(n) / math.log(1 / decay))
    # bound stated after Definition 1
    assert (bounds[:, 1] - bounds[:, 0]).sum() <= 6 * n * n_layers


def test_seeded_intervals_min_length_and_short_series():
    bounds = seeded_intervals(100, 2**-0.5, 10)
    assert np.all(bounds[:, 1] - bounds[:, 0] >= 10)
    assert seeded_intervals(1, 0.5, 2).shape == (0, 2)
    with pytest.raises(ValueError):
        seeded_intervals(100, 0.4, 2)


def test_wild_intervals():
    bounds = wild_intervals(200, 500, 4, np.random.default_rng(0))
    assert (0, 200) in _as_set(bounds)
    assert len(bounds) <= 501
    assert np.all(bounds[:, 1] - bounds[:, 0] >= 4)
    again = wild_intervals(200, 500, 4, np.random.default_rng(0))
    np.testing.assert_array_equal(bounds, again)


# --- gains and selections --------------------------------------------------


def test_l2_gain_is_the_squared_cusum():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(60) + np.repeat([0.0, 2.0], 30)
    solver = Binseg(model="l2", min_size=1, jump=1, backend="python").fit(x)
    start, end = 7, 52
    n = end - start
    cusum = []
    for s in range(start + 1, end):
        left, right = s - start, end - s
        t = (
            math.sqrt(right / (n * left)) * x[start:s].sum()
            - math.sqrt(left / (n * right)) * x[s:end].sum()
        )
        cusum.append(t)
    best = int(np.argmax(np.abs(cusum)))
    bkp, gain, _ = solver.single_bkp(start, end)
    assert bkp == start + 1 + best
    assert gain == pytest.approx(cusum[best] ** 2, rel=1e-10)


# intervals (start, end), best split, gain
_CANDIDATES = (
    np.array([0, 0, 3, 6]),
    np.array([10, 4, 8, 10]),
    np.array([5, 2, 6, 8]),
    np.array([9.0, 5.0, 7.0, 1.0]),
)


def test_greedy_path_drops_intervals_containing_a_change_point():
    # (3, 8) contains 5, selected first: dropped
    assert greedy_path(*_CANDIDATES) == [(5, 9.0), (2, 5.0), (8, 1.0)]


def test_narrowest_over_threshold():
    # narrowest first: (0, 4) -> 2, (6, 10) -> 8, (3, 8) -> 6; (0, 10) holds 2
    assert narrowest_over_threshold(*_CANDIDATES, 0.0) == [
        (2, 5.0),
        (6, 7.0),
        (8, 1.0),
    ]
    assert narrowest_over_threshold(*_CANDIDATES, 2.0) == [(2, 5.0), (6, 7.0)]


@pytest.mark.parametrize("seed", range(5))
def test_narrowest_path_matches_each_threshold(seed):
    rng = np.random.default_rng(seed)
    n = 60
    ends = rng.integers(0, n + 1, size=(80, 2))
    ends.sort(axis=1)
    ends = ends[ends[:, 1] - ends[:, 0] >= 2]
    starts, stops = ends[:, 0], ends[:, 1]
    bkps = np.array([rng.integers(s + 1, e) for s, e in ends])
    gains = rng.choice(np.arange(1.0, 30.0), size=len(ends))  # with ties
    expected = []
    for g in np.unique(gains)[::-1]:
        solution = narrowest_over_threshold(starts, stops, bkps, gains, g - 0.5)
        if not expected or solution != expected[-1]:
            expected.append(solution)
    assert list(narrowest_path(starts, stops, bkps, gains)) == expected


def test_ssic_univariate_formula():
    x = np.array([0.0, 1.0, 0.0, 5.0, 6.0, 5.0, 7.0])
    zeros = np.zeros((1, 1))
    csum = np.vstack([zeros, np.cumsum(x)[:, None]])
    csum2 = np.vstack([zeros, np.cumsum(x**2)[:, None]])
    n = len(x)
    rss = ((x[:3] - x[:3].mean()) ** 2).sum() + ((x[3:] - x[3:].mean()) ** 2).sum()
    expected = n / 2 * math.log(rss / n) + 1 * math.log(n) ** 1.01
    assert ssic(csum, csum2, [3], 1.01) == pytest.approx(expected)


# --- detector ----------------------------------------------------------------


@pytest.mark.parametrize("intervals", ["seeded", "wild"])
@pytest.mark.parametrize("selection", ["greedy", "narrowest"])
@pytest.mark.parametrize(
    "stop", [{}, {"n_cps": 3}, {"penalty": 200.0}], ids=["ssic", "n_cps", "penalty"]
)
def test_detects_steps(intervals, selection, stop):
    x = _steps([0, 3, -1, 2], 100, 1.0)
    detector = SeededBinSegDetector(
        intervals=intervals, selection=selection, random_state=0, **stop
    )
    cps = detector.fit_predict(x)
    assert len(cps) == 3
    np.testing.assert_allclose(cps, [100, 200, 300], atol=3)


def test_frequent_alternating_changes():
    # "teeth": binary segmentation's first split on the whole series sees no
    # change; the narrow seeded intervals do
    x = _steps([0, 1] * 7, 10, 0.1)
    cps = SeededBinSegDetector().fit_predict(x)
    np.testing.assert_array_equal(cps, np.arange(10, 140, 10))


@pytest.mark.parametrize("selection", ["greedy", "narrowest"])
def test_guided_returns_exactly_n_cps(selection):
    x = _steps([0, 3, -1, 2, 0], 60, 1.0)
    for k in (0, 1, 2, 4, 7):
        cps = SeededBinSegDetector(n_cps=k, selection=selection).fit_predict(x)
        assert len(cps) == k
        assert np.all(np.diff(cps) > 0)


def test_ssic_rarely_splits_noise():
    rng = np.random.default_rng(3)
    false = [
        len(SeededBinSegDetector().fit_predict(rng.standard_normal(300)))
        for _ in range(20)
    ]
    assert sum(false) <= 2


def test_multivariate_change_in_one_channel():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((300, 3))
    X[150:, 1] += 2.0
    cps = SeededBinSegDetector().fit_predict(X)
    assert len(cps) == 1 and abs(cps[0] - 150) <= 3


def test_other_cost_with_penalty():
    x = _steps([0, 3, 0], 100, 0.5)
    cps = SeededBinSegDetector(model="l1", penalty=50.0).fit_predict(x)
    np.testing.assert_allclose(cps, [100, 200], atol=3)


def test_ssic_needs_l2():
    with pytest.raises(ValueError, match="sSIC"):
        SeededBinSegDetector(model="rbf")


def test_both_backends_agree():
    x = _steps([0, 2, -1, 1], 80, 1.0, seed=5)
    python = SeededBinSegDetector(backend="python").fit_predict(x)
    auto = SeededBinSegDetector(backend="auto").fit_predict(x)
    np.testing.assert_array_equal(python, auto)


def test_wild_intervals_are_reproducible():
    x = _steps([0, 2, 0], 100, 1.0)
    first = SeededBinSegDetector(intervals="wild", n_intervals=200, random_state=1)
    second = SeededBinSegDetector(intervals="wild", n_intervals=200, random_state=1)
    np.testing.assert_array_equal(first.fit_predict(x), second.fit_predict(x))
    assert first.n_intervals_ <= 201
