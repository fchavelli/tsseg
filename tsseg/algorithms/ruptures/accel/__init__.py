"""numba backend of the vendored ruptures solvers.

``Pelt`` and ``Dynp`` hand their whole search to :func:`pelt` and :func:`dynp`;
``Binseg``, ``BottomUp`` and ``Window`` keep their Python control flow and
evaluate their segments through a :class:`Signal`. The costs covered are
``CostL1``, ``CostL2``, ``CostRbf`` and ``CostCosine`` (these exact types: a
subclass may redefine ``error``); the kernel costs never build the n x n Gram
matrix, in O(n) memory instead of O(n^2).

Both backends return the same segmentation. Costs summed in another order
than numpy's differ from the Python ones by rounding, so every decision of the
solvers treats values closer than ``utils.TIE_RTOL`` (relative) as equal and
breaks such ties by position, on both backends alike. The sweeps of l1 and l2
run on a copy of the signal whose channels far from 0 are shifted to it
(:func:`shifted`), so that their rounding stays at the scale of the costs.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import BadSegmentationParameters, NotEnoughPoints
from ..utils import tie_unit
from ._costs import COSINE, L1, L2, RBF, row_norms, segment_cost
from ._greedy import binseg_split, window_scores
from ._jit import AVAILABLE
from ._median import median_sq_dist
from ._optimal import dynp as _dynp
from ._optimal import pelt as _pelt

__all__ = [
    "AVAILABLE",
    "BACKENDS",
    "MODELS",
    "Signal",
    "dynp",
    "kind_of",
    "median_sq_dist",
    "pelt",
    "shifted",
    "use_numba",
]

BACKENDS = ("auto", "numba", "python")
#: The cost models with a numba path, by ``BaseCost.model``.
MODELS = {"l1": L1, "l2": L2, "rbf": RBF, "cosine": COSINE}


def kind_of(cost) -> int | None:
    """Code of ``cost``'s model for the numba kernels; None if they lack it."""
    from ..costs import CostCosine, CostL1, CostL2, CostRbf

    kinds = {CostL1: L1, CostL2: L2, CostRbf: RBF, CostCosine: COSINE}
    return kinds.get(type(cost))


def use_numba(cost, backend: str) -> bool:
    """Whether a solver runs ``cost`` with numba rather than in Python.

    ``"auto"`` does when numba is installed and the cost is one of ``MODELS``,
    ``"numba"`` insists (and raises otherwise), ``"python"`` never does.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "python":
        return False
    handled = kind_of(cost) is not None
    if backend == "numba":
        if not AVAILABLE:
            raise ImportError(
                "backend='numba' needs numba (pip install tsseg[accelerators])"
            )
        if not handled:
            raise ValueError(
                f"backend='numba' handles the costs {sorted(MODELS)}, "
                f"not {cost.model!r}"
            )
    return handled and AVAILABLE


def shifted(x: np.ndarray) -> np.ndarray:
    """``x`` with each channel far from 0 (|median| > 4 std) minus its median.

    The costs l1 and l2 do not change when a channel is shifted, but the sweeps
    add up the values themselves (Welford's running mean, the sums of the
    heaps): on a channel at 1e8 with a spread of 1 they would lose 8 of their
    16 digits, where numpy, which subtracts the mean or the median first, loses
    none. The shift is exact for every value within |median| / 2 of the median
    (Sterbenz), 2 std at least. Channels near 0, z-normalised ones among them
    (|median - mean| <= std), are returned as they are.
    """
    median = np.median(x, axis=0)
    shift = np.where(np.abs(median) > 4.0 * x.std(axis=0), median, 0.0)
    if not shift.any():
        return x
    return np.ascontiguousarray(x - shift)


class Signal:
    """The signal of a fitted cost, prepared for the numba kernels."""

    def __init__(self, cost) -> None:
        self.kind = kind_of(cost)
        self.min_size = cost.min_size
        #: the signal, for ``segment_cost`` (numpy's arithmetic, bit for bit)
        self.x = np.ascontiguousarray(cost.signal, dtype=np.float64)
        #: the signal of the sweeps: l1 and l2 on :func:`shifted` channels
        self.xs = shifted(self.x) if self.kind in (L1, L2) else self.x
        self.gamma = float(cost.gamma) if self.kind == RBF else 1.0
        self.norms = row_norms(self.x) if self.kind == COSINE else np.empty(0)
        #: ``utils.tie_unit`` of the cost, computed as on the Python path
        self.unit = tie_unit(cost)
        # scratch of segment_cost: the l1 and l2 costs of up to n x d values
        self._buf = np.empty(0 if self.kind in (RBF, COSINE) else self.x.size)
        self._errors: dict[tuple[int, int], float] = {}

    def error(self, start: int, end: int) -> float:
        """``cost.error(start, end)``, memoised."""
        key = (start, end)
        if key not in self._errors:
            if end - start < self.min_size:
                raise NotEnoughPoints
            self._errors[key] = float(
                segment_cost(
                    self.x, start, end, self.kind, self.gamma, self.norms, self._buf
                )
            )
        return self._errors[key]

    def split(self, start: int, end: int, min_size: int, jump: int):
        """``Binseg.single_bkp``: (change point or None, gain, segment cost)."""
        bkp, gain, whole = binseg_split(
            self.xs,
            start,
            end,
            min_size,
            jump,
            self.kind,
            self.gamma,
            self.norms,
            self.unit,
        )
        return (None if bkp < 0 else int(bkp)), float(gain), float(whole)

    def window_scores(self, inds: np.ndarray, width: int):
        """``Window``'s scores around ``inds`` and the cost of each window."""
        inds = np.asarray(inds, dtype=np.int64)
        if inds.size and width // 2 < self.min_size:
            raise NotEnoughPoints
        return window_scores(self.x, inds, width, self.kind, self.gamma, self.norms)


def pelt(cost, pen: float, min_size: int, jump: int) -> list[int]:
    """``Pelt._seg`` on a fitted cost: the segment ends."""
    signal = Signal(cost)
    ends = _pelt(
        signal.xs,
        float(pen),
        int(min_size),
        int(jump),
        signal.kind,
        signal.gamma,
        signal.unit,
    )
    if ends.size == 0:
        raise ValueError("PELT found no admissible segmentation")
    return [int(e) for e in ends]


def dynp(cost, n_bkps: int, min_size: int, jump: int) -> list[int]:
    """``Dynp.seg(0, n, n_bkps)`` on a fitted cost: the segment ends."""
    signal = Signal(cost)
    ends = _dynp(
        signal.xs,
        int(n_bkps),
        int(min_size),
        int(jump),
        signal.kind,
        signal.gamma,
        signal.unit,
    )
    if ends.size == 0:
        raise BadSegmentationParameters
    return [int(e) for e in ends]
