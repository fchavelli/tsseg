"""Dynamic programming segmentation."""

from __future__ import annotations

from functools import lru_cache

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import sanity_check


class Dynp(BaseEstimator):
    """Dynamic programming change point detection."""

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

    @lru_cache(maxsize=None)
    def seg(self, start: int, end: int, n_bkps: int):
        jump, min_size = self.jump, self.min_size
        if n_bkps == 0:
            return {(start, end): self.cost.error(start, end)}
        admissible = []
        for bkp in range(start, end):
            if bkp % jump != 0:
                continue
            if bkp - start < min_size or end - bkp < min_size:
                continue
            if not sanity_check(
                n_samples=bkp - start,
                n_bkps=n_bkps - 1,
                jump=jump,
                min_size=min_size,
            ):
                continue
            admissible.append(bkp)
        if not admissible:
            raise BadSegmentationParameters
        sub_partitions = []
        for bkp in admissible:
            left = self.seg(start, bkp, n_bkps - 1)
            right = self.seg(bkp, end, 0)
            partition = dict(left)
            partition[(bkp, end)] = right[(bkp, end)]
            sub_partitions.append(partition)
        return min(sub_partitions, key=lambda d: sum(d.values()))

    def fit(self, signal) -> "Dynp":
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        self.seg.cache_clear()
        return self

    def predict(self, n_bkps: int):
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        partition = self.seg(0, self.n_samples, n_bkps)
        return sorted(end for _, end in partition.keys())

    def fit_predict(self, signal, n_bkps: int):
        self.fit(signal)
        return self.predict(n_bkps)
