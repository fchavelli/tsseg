"""Binary segmentation algorithm."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import pairwise, sanity_check


class Binseg(BaseEstimator):
    """Binary segmentation change point detection."""

    def __init__(
        self,
        model: str = "l2",
        custom_cost: BaseCost | None = None,
        min_size: int = 2,
        jump: int = 5,
        params: dict | None = None,
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
        self.n_samples: int | None = None
        self.signal: np.ndarray | None = None

    def _seg(self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        if self.n_samples is None:
            raise RuntimeError("Estimator not fitted")
        bkps = [self.n_samples]
        stop = False
        while not stop:
            stop = True
            candidates = [self.single_bkp(start, end) for start, end in pairwise([0] + bkps)]
            bkp, gain = max(candidates, key=lambda x: x[1])
            if bkp is None:
                break
            if n_bkps is not None:
                if len(bkps) - 1 < n_bkps:
                    stop = False
            elif pen is not None:
                if gain > pen:
                    stop = False
            elif epsilon is not None:
                error = self.cost.sum_of_costs(bkps)
                if error > epsilon:
                    stop = False
            if not stop:
                bkps.append(bkp)
                bkps.sort()
        return {(start, end): self.cost.error(start, end) for start, end in pairwise([0] + bkps)}

    @lru_cache(maxsize=None)
    def single_bkp(self, start: int, end: int):
        segment_cost = self.cost.error(start, end)
        if np.isinf(segment_cost) and segment_cost < 0:
            return None, 0.0
        gains = []
        for bkp in range(start, end, self.jump):
            if bkp - start >= self.min_size and end - bkp >= self.min_size:
                gain = segment_cost - self.cost.error(start, bkp) - self.cost.error(bkp, end)
                gains.append((gain, bkp))
        if not gains:
            return None, 0.0
        gain, bkp = max(gains)
        return bkp, gain

    def fit(self, signal: np.ndarray) -> "Binseg":
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self.n_samples = self.signal.shape[0]
        self.cost.fit(signal)
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
