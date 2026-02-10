"""Penalised change point detection (PELT)."""

from __future__ import annotations

from math import floor

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import sanity_check


class Pelt(BaseEstimator):
    """PELT change point detection algorithm."""

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

    def _seg(self, pen: float):
        if self.n_samples is None:
            raise RuntimeError("Estimator not fitted")
        partitions: dict[int, dict[tuple[int, int], float]] = {0: {(0, 0): 0.0}}
        admissible: list[int] = []
        indices = [k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        indices.append(self.n_samples)
        for bkp in indices:
            new_adm = floor((bkp - self.min_size) / self.jump) * self.jump
            admissible.append(new_adm)
            subproblems = []
            for t in admissible:
                left = partitions.get(t)
                if left is None:
                    continue
                right_cost = self.cost.error(t, bkp) + pen
                tmp = left.copy()
                tmp[(t, bkp)] = right_cost
                subproblems.append(tmp)
            if not subproblems:
                continue
            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            best = partitions[bkp]
            admissible = [
                t
                for t, partition in zip(admissible, subproblems)
                if sum(partition.values()) <= sum(best.values()) + pen
            ]
        best_partition = partitions[self.n_samples]
        best_partition.pop((0, 0), None)
        return best_partition

    def fit(self, signal) -> "Pelt":
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, pen: float):
        if pen <= 0:
            raise ValueError("pen must be positive")
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        partition = self._seg(pen)
        return sorted(end for _, end in partition.keys())

    def fit_predict(self, signal, pen: float):
        self.fit(signal)
        return self.predict(pen)
