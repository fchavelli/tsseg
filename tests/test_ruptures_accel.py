"""The numba backend of the vendored ruptures (``ruptures.accel``) against the
Python path.

``Pelt``, ``Dynp``, ``Binseg``, ``BottomUp`` and ``Window`` run the costs
``l1``, ``l2``, ``rbf`` and ``cosine`` either in Python (``cost.error``, the
kernel costs over the n x n Gram matrix) or with numba, in O(n) memory. Both
backends must return the same segmentation.
"""

from itertools import combinations, product

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from tsseg.algorithms.binseg.detector import BinSegDetector
from tsseg.algorithms.bottomup.detector import BottomUpDetector
from tsseg.algorithms.dynp.detector import DynpDetector
from tsseg.algorithms.kcpd.detector import KCPDDetector
from tsseg.algorithms.pelt.detector import PeltDetector
from tsseg.algorithms.ruptures import accel
from tsseg.algorithms.ruptures.costs import CostCosine, CostL1, CostL2, CostRbf
from tsseg.algorithms.ruptures.detection import Binseg, BottomUp, Dynp, Pelt, Window
from tsseg.algorithms.ruptures.exceptions import (
    BadSegmentationParameters,
    NotEnoughPoints,
)
from tsseg.algorithms.window.detector import WindowDetector

pytestmark = pytest.mark.skipif(not accel.AVAILABLE, reason="needs numba")

BACKENDS = ("python", "numba")
MODELS = ("l1", "l2", "rbf", "cosine")
COSTS = {"l1": CostL1, "l2": CostL2, "rbf": CostRbf, "cosine": CostCosine}


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


def _both(algo, x, init, **predict):
    """``algo(**init, backend=b).fit(x).predict(**predict)`` on both backends,
    or the name of the exception raised."""
    out = []
    for backend in BACKENDS:
        try:
            out.append(algo(**init, backend=backend).fit(x).predict(**predict))
        except (BadSegmentationParameters, NotEnoughPoints, ValueError) as exc:
            out.append(type(exc).__name__)
    return out


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (5, 3), (8, 4)])
@pytest.mark.parametrize("pen", [0.5, 5.0])
def test_pelt_backends_agree(model, min_size, jump, pen):
    for seed in range(3):
        x = _signal(seed)
        a, b = _both(
            Pelt,
            x,
            {"model": model, "min_size": min_size, "jump": jump},
            pen=pen,
        )
        assert a == b


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (5, 3)])
@pytest.mark.parametrize("n_bkps", [1, 4])
def test_dynp_backends_agree(model, min_size, jump, n_bkps):
    for seed in range(2):
        x = _signal(seed, n=120)
        a, b = _both(
            Dynp,
            x,
            {"model": model, "min_size": min_size, "jump": jump},
            n_bkps=n_bkps,
        )
        assert a == b


CRITERIA = [{"n_bkps": 3}, {"pen": 2.0}, {"pen": 20.0}, {"epsilon": 50.0}]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (5, 3)])
@pytest.mark.parametrize("criterion", CRITERIA, ids=str)
def test_binseg_backends_agree(model, min_size, jump, criterion):
    for seed in range(2):
        x = _signal(seed, n=160)
        a, b = _both(
            Binseg,
            x,
            {"model": model, "min_size": min_size, "jump": jump},
            **criterion,
        )
        assert a == b


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("min_size,jump", [(1, 1), (2, 5), (5, 3)])
@pytest.mark.parametrize("criterion", CRITERIA, ids=str)
def test_bottomup_backends_agree(model, min_size, jump, criterion):
    for seed in range(2):
        x = _signal(seed, n=160)
        a, b = _both(
            BottomUp,
            x,
            {"model": model, "min_size": min_size, "jump": jump},
            **criterion,
        )
        assert a == b


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("width,jump", [(10, 1), (20, 5), (41, 3)])
@pytest.mark.parametrize("criterion", CRITERIA, ids=str)
def test_window_backends_agree(model, width, jump, criterion):
    for seed in range(2):
        x = _signal(seed, n=160)
        a, b = _both(
            Window,
            x,
            {"width": width, "model": model, "jump": jump},
            **criterion,
        )
        assert a == b


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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("min_size,jump", list(product([1, 2, 3], [1, 2, 3])))
def test_optimal_solvers_reach_the_optimum(model, min_size, jump):
    """Pelt and Dynp, both backends, against an exhaustive enumeration."""
    rng = np.random.default_rng(1)
    for n in (6, 9, 12):
        for x in rng.normal(size=(3, n, 2)):
            cost = COSTS[model]().fit(x)
            size = max(min_size, cost.min_size)
            feasible = _feasible(n, size, jump)
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


def _plateaus(d=2):
    """Constant stretches: the costs inside them are 0 (or rounding residues),
    so that many segmentations tie exactly."""
    levels = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.3, 0.7]])[:, :d]
    return np.repeat(levels, [40, 30, 50, 20], axis=0)


@pytest.mark.parametrize("model", MODELS)
def test_exact_ties_are_broken_alike(model):
    x = _plateaus()
    for min_size, jump in ((2, 1), (5, 5), (3, 2)):
        for n_bkps in (3, 6):
            a, b = _both(
                Dynp,
                x,
                {"model": model, "min_size": min_size, "jump": jump},
                n_bkps=n_bkps,
            )
            assert a == b
        a, b = _both(
            Pelt,
            x,
            {"model": model, "min_size": min_size, "jump": jump},
            pen=0.01,
        )
        assert a == b
        for criterion in ({"n_bkps": 6}, {"pen": 0.01}):
            for algo in (Binseg, BottomUp):
                a, b = _both(
                    algo,
                    x,
                    {"model": model, "min_size": min_size, "jump": jump},
                    **criterion,
                )
                assert a == b
            a, b = _both(
                Window,
                x,
                {"width": 12, "model": model, "jump": jump},
                **criterion,
            )
            assert a == b


def test_pelt_prunes_alike_at_the_bound():
    """A start whose value equals the pruning bound up to rounding (Python's
    sum is compensated, the numba one is not) is kept on both backends. Here
    the Python path used to prune a start that numba kept and that later won a
    tie: two segmentations of equal cost."""
    rng = np.random.default_rng(1387)
    n, d = int(rng.integers(40, 260)), int(rng.integers(1, 4))
    x = rng.integers(0, 3, (n, d)).astype(float)
    pen = 0.05 * float(np.std(x))
    a, b = _both(Pelt, x, {"model": "l1", "min_size": 4, "jump": 4}, pen=pen)
    assert a == b


@pytest.mark.parametrize("model", ["l1", "l2"])
def test_ties_are_scale_invariant(model):
    """Scaling the signal by a power of 2 scales every cost exactly, so the
    segmentation must not move; the tie gap follows the costs' scale."""
    x = _signal(2, n=150)
    power = 2 if model == "l2" else 1
    for backend in BACKENDS:
        results = []
        for scale in (2.0**-24, 1.0, 2.0**16):
            y = x * scale
            pen = 3.0 * scale**power
            results.append(
                (
                    Dynp(model=model, min_size=2, jump=1, backend=backend)
                    .fit(y)
                    .predict(4),
                    Pelt(model=model, min_size=2, jump=1, backend=backend)
                    .fit(y)
                    .predict(pen),
                    Binseg(model=model, min_size=2, jump=1, backend=backend)
                    .fit(y)
                    .predict(pen=pen),
                    BottomUp(model=model, min_size=2, jump=1, backend=backend)
                    .fit(y)
                    .predict(pen=pen),
                    Window(width=20, model=model, jump=1, backend=backend)
                    .fit(y)
                    .predict(pen=pen),
                )
            )
        assert results[0] == results[1] == results[2]


def _all_algos(x, model, backend):
    """The five solvers with a change point count and a penalty."""
    init = {"model": model, "min_size": 2, "jump": 1, "backend": backend}
    return (
        Pelt(**init).fit(x).predict(3.0),
        Dynp(**init).fit(x).predict(3),
        Binseg(**init).fit(x).predict(n_bkps=3),
        Binseg(**init).fit(x).predict(pen=10.0),
        BottomUp(**init).fit(x).predict(n_bkps=3),
        BottomUp(**init).fit(x).predict(pen=10.0),
        Window(width=40, **init).fit(x).predict(n_bkps=3),
        Window(width=40, **init).fit(x).predict(pen=10.0),
    )


@pytest.mark.parametrize("model", ["l1", "l2"])
def test_signals_far_from_zero(model):
    """A signal at 1e8 with a spread of 1 segments as the same signal at 0.
    The sweeps would lose 8 digits summing the values themselves (l1 then
    differed between the backends): they run on the signal shifted by its
    median. And the tie gap stays at the scale of the costs: a gap following
    the magnitude of x^2 tied most l2 candidates."""
    rng = np.random.default_rng(0)
    for seed in (0, 1):
        steps = np.repeat(rng.integers(0, 4, (4, 2)), [60, 50, 70, 60], axis=0)
        for base in (
            _signal(seed, n=240, d=2),
            steps + rng.integers(0, 3, (240, 2)),  # integers: exact ties
        ):
            for offset in (1e8, 1e8 + 0.1234567):
                expected = _all_algos(base, model, "python")
                for backend in BACKENDS:
                    assert _all_algos(base + offset, model, backend) == expected


def test_signals_are_costed_in_float64():
    """float32 and integer signals are converted by the costs' ``fit``: numpy
    summed float32 in float32, where numba sums in float64."""
    x = _signal(4, n=200)
    for model in MODELS:
        for y in ((x * 1.37).astype(np.float32), np.round(x * 3).astype(np.int64)):
            assert _all_algos(y, model, "python") == _all_algos(y, model, "numba")


def test_dynp_backends_raise_alike():
    """``Dynp`` raises from inside its recursion when a sub-problem has no
    admissible change point; the numba solver must refuse the same inputs."""
    x = _signal(0, n=30, n_cps=1)
    for model in MODELS:
        for n_bkps, min_size, jump in [(5, 5, 5), (3, 4, 7), (2, 9, 3)]:
            a, b = _both(
                Dynp,
                x,
                {"model": model, "min_size": min_size, "jump": jump},
                n_bkps=n_bkps,
            )
            assert a == b


def test_window_backends_raise_alike():
    """A window too short for the cost's ``min_size`` (2 for l1)."""
    x = _signal(0, n=60)
    a, b = _both(Window, x, {"width": 2, "model": "l1"}, n_bkps=2)
    assert a == b == "NotEnoughPoints"


@pytest.mark.parametrize("model", ["l1", "l2"])
def test_segment_costs_are_numpy_bit_for_bit(model):
    """On a C-contiguous float64 signal, the numba costs of one segment sum in
    numpy's order (pairwise by buffers of 8192 values) and equal the Python
    ones exactly."""
    rng = np.random.default_rng(3)
    for d, n in ((1, 20000), (3, 500), (9, 1200)):
        x = rng.normal(rng.normal(0, 3), 2.0, (n, d))
        cost = COSTS[model]().fit(x)
        signal = accel.Signal(cost)
        for start, end in [(0, n), (5, 17), (1, n - 1)] + [
            tuple(sorted(rng.choice(n + 1, 2, replace=False))) for _ in range(20)
        ]:
            if end - start >= cost.min_size:
                assert signal.error(start, end) == cost.error(start, end)


@pytest.mark.parametrize("model", MODELS)
def test_sweeps_match_segment_costs(model):
    """The costs from sweeps (Binseg's splits) agree with ``cost.error`` up to
    rounding."""
    x = _signal(4, n=90)
    cost = COSTS[model]().fit(x)
    signal = accel.Signal(cost)
    for start, end in ((0, 90), (10, 50), (33, 90)):
        bkp, gain, whole = signal.split(start, end, 2, 1)
        assert whole == pytest.approx(cost.error(start, end), rel=1e-12, abs=1e-12)
        expected = whole - cost.error(start, bkp) - cost.error(bkp, end)
        assert gain == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("model", ["rbf", "cosine"])
def test_kernel_costs_never_build_the_gram(model):
    x = _signal(5, n=200)
    algos = [
        Pelt(model=model, backend="numba"),
        Dynp(model=model, backend="numba"),
        Binseg(model=model, backend="numba"),
        BottomUp(model=model, backend="numba"),
        Window(width=20, model=model, backend="numba"),
    ]
    for algo, kwargs in zip(
        algos, [{"pen": 1.0}, {"n_bkps": 3}, {"n_bkps": 3}, {"n_bkps": 3}, {"pen": 1.0}]
    ):
        algo.fit(x).predict(**kwargs)
        assert algo.cost._gram is None


def test_detectors_backends_agree():
    x = _signal(3, n=400, d=2)
    for params in (
        {"model": "rbf", "penalty": 5.0, "min_size": 2, "jump": 5},
        {"model": "l1", "penalty": 5.0, "min_size": 3, "jump": 2},
        {"model": "l2", "penalty": 20.0, "min_size": 2, "jump": 5},
    ):
        cps = [PeltDetector(**params, backend=b).fit_predict(x) for b in BACKENDS]
        np.testing.assert_array_equal(*cps)
    for model in MODELS:
        cps = [
            DynpDetector(model=model, n_cps=4, jump=5, backend=b).fit_predict(x, None)
            for b in BACKENDS
        ]
        np.testing.assert_array_equal(*cps)
        for Detector in (BinSegDetector, BottomUpDetector):
            cps = [
                Detector(model=model, penalty=5.0, backend=b).fit_predict(x)
                for b in BACKENDS
            ]
            np.testing.assert_array_equal(*cps)
        cps = [
            WindowDetector(model=model, width=40, pen=5.0, backend=b).fit_predict(x)
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
        assert accel.median_sq_dist(x) == pytest.approx(expected, rel=1e-12)
    assert accel.median_sq_dist(np.ones((50, 2))) == 0.0


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


def test_backend_numba_needs_a_supported_cost():
    x = _signal(0, n=60)
    for model in ("normal", "linear"):
        with pytest.raises(ValueError, match="backend='numba'"):
            Pelt(model=model, backend="numba").fit(x).predict(1.0)
    with pytest.raises(ValueError, match="backend must be"):
        Pelt(model="rbf", backend="c").fit(x).predict(1.0)
    with pytest.raises(ValueError, match="backend must be"):
        Binseg(model="l2", backend="c").fit(x)


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
    expected = {
        algo: algo(model="l1", backend="numba").fit(x).predict(pen=2.0)
        for algo in (Pelt, Binseg, BottomUp)
    }
    monkeypatch.setattr(accel, "AVAILABLE", False)
    for algo, bkps in expected.items():
        assert algo(model="l1").fit(x).predict(pen=2.0) == bkps
    with pytest.raises(ImportError, match="needs numba"):
        Pelt(model="rbf", backend="numba").fit(x).predict(2.0)
