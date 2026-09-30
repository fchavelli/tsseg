"""Binary segmentation algorithm."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import pairwise, sanity_check, tie_tol, tie_unit


class Binseg(BaseEstimator):
    """Binary segmentation change point detection.

    ``backend``: ``"auto"`` computes the splits of the costs ``l1``, ``l2``,
    ``rbf`` and ``cosine`` with numba (``ruptures.accel``) when it is installed:
    the costs of all the candidate halves of a segment in one sweep from each
    end, instead of two ``error`` calls per candidate, and no Gram matrix.
    ``"numba"`` and ``"python"`` force either path; both return the same change
    points.

    Gains closer than the tie gap of their segments' costs (``utils.tie_tol``)
    are equal: a segment's best change point is then its last tied one, and
    the segment split is the first tied one, as ``max`` picks among exact
    ties; a gain tied with ``pen`` stops.
    """

    def __init__(
        self,
        model: str = "l2",
        custom_cost: BaseCost | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict | None = None,
        backend: str = "auto",
    ) -> None:
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.backend = backend
        self.n_samples: int | None = None
        self.signal: np.ndarray | None = None
        self._fast = None

    def _seg(self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        if self.n_samples is None:
            raise RuntimeError("Estimator not fitted")
        bkps = [self.n_samples]
        stop = False
        while not stop:
            stop = True
            segments = list(pairwise([0] + bkps))
            candidates = [self.single_bkp(start, end) for start, end in segments]
            top = max(range(len(candidates)), key=lambda i: candidates[i][1])
            i = next(
                i
                for i, (_, gain, _) in enumerate(candidates)
                if gain >= candidates[top][1] - self._tol(segments[i], segments[top])
            )
            bkp, gain, _ = candidates[i]
            if bkp is None:
                break
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if gain > pen + self._tol(segments[i]):
                    stop = False
            elif epsilon is not None:
                error = sum(cost for _, _, cost in candidates)  # sum_of_costs(bkps)
                if error > epsilon + tie_tol(error, self.n_samples * self._unit):
                    stop = False
            if not stop:
                bkps.append(bkp)
                bkps.sort()
        return {(start, end): self.single_bkp(start, end)[2] for start, end in pairwise([0] + bkps)}

    def _tol(self, *segments: tuple[int, int]) -> float:
        """Tie gap of gains computed from these segments' costs."""
        costs = [self.single_bkp(start, end)[2] for start, end in segments]
        lengths = [end - start for start, end in segments]
        return tie_tol(*costs, max(lengths) * self._unit)

    @lru_cache(maxsize=None)
    def single_bkp(self, start: int, end: int):
        """Best change point of [start, end) (None if none is admissible), its
        gain, and the cost of [start, end)."""
        if self._fast is not None:
            return self._fast.split(start, end, self.min_size, self.jump)
        segment_cost = self.cost.error(start, end)
        if np.isinf(segment_cost) and segment_cost < 0:
            return None, 0.0, segment_cost
        gains = []
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = segment_cost - self.cost.error(start, bkp) - self.cost.error(bkp, end)
                gains.append((gain, bkp))
        if not gains:
            return None, 0.0, segment_cost
        limit = max(gains)[0] - tie_tol(segment_cost, (end - start) * self._unit)
        bkp, gain = max((bkp, gain) for gain, bkp in gains if gain >= limit)
        return bkp, gain, segment_cost

    def fit(self, signal: np.ndarray) -> "Binseg":
        from .. import accel  # numba, imported on first use

        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples = self.signal.shape[0]
        self.cost.fit(signal)
        self._unit = tie_unit(self.cost)
        self._fast = accel.Signal(self.cost) if accel.use_numba(self.cost, self.backend) else None
        self.single_bkp.cache_clear()
        return self

    def predict(self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        partition = self._seg(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
        return sorted(end for _, end in partition.keys())

    def fit_predict(self, signal: np.ndarray, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
