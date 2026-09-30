"""Vendored ``Window``: peaks are searched as upstream ruptures searches them,
``scipy.signal.argrelmax(score, order, mode="wrap")``.

The vendored ``argrelmax_1d`` used to cut the neighbourhood of a score at the
ends of the array, where upstream wraps around: a score falling away from the
first admissible index was a peak, and ``Window`` returned a change point
there that upstream does not.
"""

import numpy as np
import pytest
from scipy.signal import argrelmax

from tsseg.algorithms.ruptures.detection import Window
from tsseg.algorithms.ruptures.utils import argrelmax_1d


def test_argrelmax_1d_is_scipy_wrap():
    rng = np.random.default_rng(0)
    for trial in range(2000):
        n = int(rng.integers(0, 40))
        if trial % 2:
            values = rng.integers(0, 5, n).astype(float)  # plateaus
        else:
            values = rng.normal(size=n)
        order = int(rng.integers(1, 50))  # up to beyond n: no maximum
        expected = argrelmax(values, order=order, mode="wrap")[0]
        assert np.array_equal(argrelmax_1d(values, order), expected)


def test_argrelmax_1d_tolerance_makes_plateaus():
    values = np.array([0.0, 1.0, 1.0 + 1e-12, 0.0, 0.5, 0.0])
    assert list(argrelmax_1d(values, 1)) == [2, 4]
    assert list(argrelmax_1d(values, 1, tol=np.full(6, 1e-9))) == [4]


@pytest.mark.parametrize("backend", ["python", "auto"])
def test_no_change_point_at_the_edge(backend):
    """Change points at 94, 100, 142 and 176; the old peak search added one at
    20, the first admissible index. Expected: ruptures 1.1.10."""
    rng = np.random.default_rng(1)
    cps = np.sort(rng.choice(np.arange(15, 185), 4, replace=False))
    x = np.concatenate(
        [
            rng.normal(rng.normal(0, 2, 3), 1.0, (end - start, 3))
            for start, end in zip([0, *cps], [*cps, 200])
        ]
    )
    x = (x - x.mean(0)) / x.std(0)
    algo = Window(width=40, model="l2", min_size=1, jump=1, backend=backend).fit(x)
    assert algo.predict(n_bkps=4) == [100, 143, 176, 200]
    assert algo.predict(pen=2.0) == [100, 143, 176, 200]
