"""Penalised change point detection (PELT)."""

from __future__ import annotations

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import sanity_check, tie_limit, tie_unit


class Pelt(BaseEstimator):
    """PELT change point detection algorithm.

    ``backend``: ``"auto"`` runs the costs ``l1``, ``l2``, ``rbf`` and ``cosine``
    with the numba solver of ``ruptures.accel`` when numba is installed (the
    kernel costs then in O(n) memory instead of the Gram matrix), and every
    other cost in Python; ``"numba"`` and ``"python"`` force either path. Both
    return the same segmentation.
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

    def _seg(self, pen: float):
        if self.n_samples is None:
            raise RuntimeError("Estimator not fitted")
        partitions: dict[int, dict[tuple[int, int], float]] = {0: {(0, 0): 0.0}}
        admissible: list[int] = []
        # A start t dominated at endpoint s can only be replaced by a segment
        # starting at s, which is legal from s + min_size on: t is pruned from
        # there, not at once (fix of upstream ruptures PR #383; pruning at s
        # returned sub-optimal segmentations whenever min_size > 1).
        prune_at: dict[int, int] = {}
        endpoints = [k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
        endpoints.append(self.n_samples)
        starts = iter([0] + endpoints[:-1])
        next_start = next(starts, None)
        for bkp in endpoints:
            while next_start is not None and next_start <= bkp - self.min_size:
                admissible.append(next_start)
                next_start = next(starts, None)
            admissible = [t for t in admissible if t not in prune_at or bkp < prune_at[t]]
            subproblems = []
            for t in admissible:
                tmp = partitions[t].copy()
                tmp[(t, bkp)] = self.cost.error(t, bkp) + pen
                subproblems.append((t, tmp, sum(tmp.values())))
            limit = tie_limit(
                min(value for _, _, value in subproblems), bkp * self._unit
            )
            _, partitions[bkp], best_value = next(
                item for item in subproblems if item[2] <= limit
            )
            # A start tied with the bound is kept: pruning less never loses the
            # optimum, and values summed in two orders (Python's sum is
            # compensated, the numba one is not) prune alike.
            bound = tie_limit(best_value + pen, bkp * self._unit)
            for t, _, value in subproblems:
                if value > bound:
                    prune_at.setdefault(t, bkp + self.min_size)
        best_partition = partitions[self.n_samples]
        best_partition.pop((0, 0), None)
        return best_partition

    def fit(self, signal) -> "Pelt":
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        self._unit = tie_unit(self.cost)
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
        from .. import accel  # numba, imported on first use

        if accel.use_numba(self.cost, self.backend):
            return accel.pelt(self.cost, pen, self.min_size, self.jump)
        partition = self._seg(pen)
        return sorted(end for _, end in partition.keys())

    def fit_predict(self, signal, pen: float):
        self.fit(signal)
        return self.predict(pen)
