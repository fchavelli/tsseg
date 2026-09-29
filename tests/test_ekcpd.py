"""Kernel costs: the numba solvers of ``ruptures.ekcpd`` against the Python path.

``Pelt`` and ``Dynp`` run ``CostRbf`` / ``CostCosine`` either in Python, over the
n x n Gram matrix, or with the numba transcription of ruptures' C solvers, in
O(n) memory. Both paths must return the same segmentation.
"""

from itertools import combinations, product

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from tsseg.algorithms.dynp.detector import DynpDetector
from tsseg.algorithms.kcpd.detector import KCPDDetector
from tsseg.algorithms.pelt.detector import PeltDetector
from tsseg.algorithms.ruptures import ekcpd
from tsseg.algorithms.ruptures.costs import CostCosine, CostRbf
from tsseg.algorithms.ruptures.detection import Dynp, Pelt
from tsseg.algorithms.ruptures.exceptions import BadSegmentationParameters

pytestmark = pytest.mark.skipif(not ekcpd.AVAILABLE, reason="needs numba")

BACKENDS = ("python", "numba")


def _signal(seed, n=240, d=3, n_cps=4):
    """Piecewise-Gaussian segments, z-normalised per channel."""
    rng = np.random.default_rng(seed)
    cps = np.sort(rng.choice(np.arange(10, n - 10), n_cps, replace=False))
    bounds = zip([0, *cps], [*cps, n])
    x = np.concatenate(
        [
            rng.normal(rng.normal(0, 2, d), 1.0, (end - start, d))
            for start, end in bounds
        ]
    )
    return (x - x.mean(0)) / x.std(0)


@pytest.mark.parametrize("model", ["rbf", "cosine"])
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (5, 3), (8, 4)])
@pytest.mark.parametrize("pen", [0.5, 5.0])
def test_pelt_backends_agree(model, min_size, jump, pen):
    for seed in range(3):
        x = _signal(seed)
        ends = [
            Pelt(model=model, min_size=min_size, jump=jump, backend=b)
            .fit(x)
            .predict(pen)
            for b in BACKENDS
        ]
        assert ends[0] == ends[1]


@pytest.mark.parametrize("model", ["rbf", "cosine"])
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (5, 3)])
@pytest.mark.parametrize("n_bkps", [1, 4])
def test_dynp_backends_agree(model, min_size, jump, n_bkps):
    for seed in range(2):
        x = _signal(seed, n=120)
        ends = [
            Dynp(model=model, min_size=min_size, jump=jump, backend=b)
            .fit(x)
            .predict(n_bkps)
            for b in BACKENDS
        ]
        assert ends[0] == ends[1]


def _feasible(n, min_size, jump):
    """Every segmentation with change points on the jump grid, as its ends."""
    grid = range(jump, n, jump)
    return [
        ends
        for count in range(len(grid) + 1)
        for internal in combinations(grid, count)
        for ends in [internal + (n,)]
        if all(e - s >= min_size for s, e in zip((0,) + internal, ends))
    ]


@pytest.mark.parametrize("model", ["rbf", "cosine"])
@pytest.mark.parametrize("min_size,jump", list(product([1, 2, 3], [1, 2, 3])))
def test_kernel_solvers_reach_the_optimum(model, min_size, jump):
    """Both backends against an exhaustive enumeration of the segmentations."""
    rng = np.random.default_rng(1)
    for n in (6, 9, 12):
        feasible = _feasible(n, min_size, jump)
        for x in rng.normal(size=(3, n, 2)):
            cost = (CostRbf() if model == "rbf" else CostCosine()).fit(x)
            costs = {ends: cost.sum_of_costs(list(ends)) for ends in feasible}
            for pen in (0.05, 0.5):
                # ruptures charges the penalty once per segment.
                value = {ends: c + pen * len(ends) for ends, c in costs.items()}
                for backend in BACKENDS:
                    algo = Pelt(
                        model=model, min_size=min_size, jump=jump, backend=backend
                    )
                    found = tuple(algo.fit(x).predict(pen))
                    assert value[found] == pytest.approx(min(value.values()), abs=1e-9)
            for n_bkps in (1, 2):
                with_k = [c for ends, c in costs.items() if len(ends) == n_bkps + 1]
                for backend in BACKENDS:
                    algo = Dynp(
                        model=model, min_size=min_size, jump=jump, backend=backend
                    )
                    try:
                        found = tuple(algo.fit(x).predict(n_bkps))
                    except BadSegmentationParameters:
                        assert not with_k
                        continue
                    assert costs[found] == pytest.approx(min(with_k), abs=1e-9)


@pytest.mark.parametrize("model", ["rbf", "cosine"])
def test_exact_ties_are_broken_alike(model):
    """Inside a constant stretch, extra change points cost nothing: many
    segmentations tie exactly, and both paths must take the same one."""
    x = np.repeat([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [40, 30, 50], axis=0)
    for min_size, jump in ((2, 1), (5, 5), (3, 2)):
        for n_bkps in (3, 5):
            ends = [
                Dynp(model=model, min_size=min_size, jump=jump, backend=b)
                .fit(x)
                .predict(n_bkps)
                for b in BACKENDS
            ]
            assert ends[0] == ends[1]
        ends = [
            Pelt(model=model, min_size=min_size, jump=jump, backend=b)
            .fit(x)
            .predict(0.01)
            for b in BACKENDS
        ]
        assert ends[0] == ends[1]


def test_dynp_backends_raise_alike():
    """``Dynp`` raises from inside its recursion when a sub-problem has no
    admissible change point; the numba solver must refuse the same inputs."""
    x = _signal(0, n=30, n_cps=1)
    for n_bkps, min_size, jump in [(5, 5, 5), (3, 4, 7), (2, 9, 3)]:
        outcomes = []
        for backend in BACKENDS:
            algo = Dynp(model="rbf", min_size=min_size, jump=jump, backend=backend)
            try:
                outcomes.append(algo.fit(x).predict(n_bkps))
            except BadSegmentationParameters:
                outcomes.append("raised")
        assert outcomes[0] == outcomes[1]


def test_detectors_backends_agree():
    x = _signal(3, n=400, d=2)
    for params in (
        {"model": "rbf", "penalty": 5.0, "min_size": 2, "jump": 5},
        {"model": "cosine", "penalty": 1.0, "min_size": 3, "jump": 2},
    ):
        cps = [PeltDetector(**params, backend=b).fit_predict(x) for b in BACKENDS]
        np.testing.assert_array_equal(*cps)
    cps = [
        DynpDetector(model="rbf", n_cps=4, jump=5, backend=b).fit_predict(x, None)
        for b in BACKENDS
    ]
    np.testing.assert_array_equal(*cps)
    for params in ({"pen": 5.0}, {"n_cps": 4, "pen": None}):
        cps = [
            KCPDDetector(
                kernel="rbf", decimation=2, min_size=4, jump=2, backend=b, **params
            ).fit_predict(x)
            for b in BACKENDS
        ]
        np.testing.assert_array_equal(*cps)


def test_median_sq_dist_matches_pdist():
    rng = np.random.default_rng(0)
    for x in (
        _signal(1, n=500),
        np.round(_signal(2, n=400)),
        rng.normal(size=(301, 1)),
    ):
        dist2 = pdist(x, "sqeuclidean")
        expected = np.median(dist2[dist2 > 0])
        assert ekcpd.median_sq_dist(x) == pytest.approx(expected, rel=1e-12)
    assert ekcpd.median_sq_dist(np.ones((50, 2))) == 0.0


def test_rbf_gamma_without_gram():
    """The median heuristic is set by ``fit``, the Gram matrix only by ``error``;
    distances of a sample to itself do not enter the median."""
    x = _signal(0, n=300)
    cost = CostRbf().fit(x)
    assert cost._gram is None
    dist2 = pdist(x, "sqeuclidean")
    assert cost.gamma == pytest.approx(1.0 / np.median(dist2[dist2 > 0]), rel=1e-12)
    cost.error(0, 10)
    assert np.all(np.diag(cost.gram) == 1.0)


def test_backend_numba_needs_a_kernel_cost():
    x = _signal(0, n=60)
    with pytest.raises(ValueError, match="backend='numba'"):
        Pelt(model="l2", backend="numba").fit(x).predict(1.0)
    with pytest.raises(ValueError, match="backend must be"):
        Pelt(model="rbf", backend="c").fit(x).predict(1.0)


def test_kcpd_is_deprecated_and_is_pelt_or_dynp():
    x = _signal(4, n=300, d=2)
    with pytest.warns(FutureWarning, match="KCPDDetector is deprecated"):
        kcpd = KCPDDetector(kernel="rbf", pen=3.0, min_size=3, jump=2)
    pelt = PeltDetector(model="rbf", penalty=3.0, min_size=3, jump=2)
    np.testing.assert_array_equal(kcpd.fit_predict(x), pelt.fit_predict(x))
    with pytest.warns(FutureWarning):
        kcpd = KCPDDetector(kernel="linear", n_cps=3, pen=None, min_size=3, jump=2)
    dynp = DynpDetector(model="l2", n_cps=3, min_size=3, jump=2)
    np.testing.assert_array_equal(kcpd.fit_predict(x), dynp.fit_predict(x, None))


def test_without_numba_auto_falls_back_to_python(monkeypatch):
    x = _signal(5, n=150)
    expected = Pelt(model="rbf", backend="numba").fit(x).predict(2.0)
    monkeypatch.setattr(ekcpd, "AVAILABLE", False)
    assert Pelt(model="rbf").fit(x).predict(2.0) == expected
    with pytest.raises(ImportError, match="needs numba"):
        Pelt(model="rbf", backend="numba").fit(x).predict(2.0)
