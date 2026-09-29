"""Vendored ``Pelt``: pruning must respect ``min_size`` (upstream ruptures PR #383).

A start dominated at endpoint ``s`` was pruned at once, although the segment
that replaces it can only start at ``s`` once ``min_size`` samples follow. PELT
then returned sub-optimal segmentations. These tests pin the counterexamples
and compare against an exhaustive enumeration of the feasible partitions.
"""

from itertools import combinations, product

import numpy as np
import pytest

from tsseg.algorithms.ruptures.detection import Pelt


@pytest.mark.parametrize(
    "values,min_size,jump,expected",
    [
        ([0, 1, 0, 0, 1], 2, 1, [5]),
        ([0, 1, 0, 0, 1], 2, 2, [5]),
        ([0, 0, 1, 1, 2, 2, 2, 0], 3, 1, [3, 8]),
        ([3, 8, 7, 0, 4, 9, 6, 0, 8, 8], 3, 1, [5, 10]),
        ([0, 9, 2, 8, 3, 0, 10], 3, 1, [3, 7]),
    ],
)
def test_counterexamples(values, min_size, jump, expected):
    signal = np.asarray(values, dtype=float)
    algo = Pelt(model="l2", min_size=min_size, jump=jump)
    assert algo.fit_predict(signal, pen=0.1) == expected


def test_only_one_segment_is_feasible():
    assert Pelt(min_size=3, jump=5).fit_predict(np.zeros(3), pen=1.0) == [3]
    assert Pelt(min_size=3, jump=1).fit_predict(np.zeros(4), pen=1.0) == [4]


@pytest.mark.parametrize("min_size,jump", list(product([1, 2, 3, 4], [1, 2, 3, 5])))
def test_matches_exhaustive_enumeration(min_size, jump):
    rng = np.random.default_rng(0)
    for n in (5, 8, 11):
        grid = range(jump, n, jump)
        feasible = [
            ends
            for count in range(len(grid) + 1)
            for internal in combinations(grid, count)
            for ends in [internal + (n,)]
            if all(e - s >= min_size for s, e in zip((0,) + internal, ends))
        ]
        for signal in rng.integers(-2, 3, size=(4, n, 2)).astype(float):
            algo = Pelt(model="l2", min_size=min_size, jump=jump).fit(signal)
            for pen in (0.1, 1.0):
                # ruptures charges the penalty once per segment.
                value = {
                    ends: algo.cost.sum_of_costs(list(ends)) + pen * len(ends)
                    for ends in feasible
                }
                found = tuple(algo.predict(pen))
                assert found in value
                assert value[found] == pytest.approx(min(value.values()), abs=1e-12)
