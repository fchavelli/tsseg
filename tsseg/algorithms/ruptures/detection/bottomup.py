"""Bottom-up segmentation algorithm."""

from __future__ import annotations

import heapq
from bisect import bisect_left
from functools import lru_cache

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import Bnode, pairwise, sanity_check


class BottomUp(BaseEstimator):
    """Bottom-up agglomerative segmentation."""

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
        self.signal = None
        self.leaves: list[Bnode] | None = None

    def _grow_tree(self) -> list[Bnode]:
        if self.n_samples is None:
            raise RuntimeError("Estimator not fitted")
        partition = [(-self.n_samples, (0, self.n_samples))]
        while partition:
            _, (start, end) = partition[0]
            mid = (start + end) * 0.5
            bkps = [
                b
                for b in range(start, end)
                if b % self.jump == 0 and b - start >= self.min_size and end - b >= self.min_size
            ]
            if not bkps:
                break
            bkp = min(bkps, key=lambda x: abs(x - mid))
            heapq.heappop(partition)
            heapq.heappush(partition, (-bkp + start, (start, bkp)))
            heapq.heappush(partition, (-end + bkp, (bkp, end)))
        partition.sort(key=lambda item: item[1])
        leaves = []
        for _, (start, end) in partition:
            cost = self.cost.error(start, end)
            leaves.append(Bnode(start, end, cost))
        return leaves

    @lru_cache(maxsize=None)
    def merge(self, left: Bnode, right: Bnode) -> Bnode:
        if left.end != right.start:
            raise ValueError("Segments must be contiguous")
        start, end = left.start, right.end
        val = self.cost.error(start, end)
        return Bnode(start, end, val, left=left, right=right)

    def _seg(self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        leaves = sorted(self.leaves)
        keys = [leaf.start for leaf in leaves]
        removed: set[Bnode] = set()
        merged: list[tuple[float, Bnode]] = []
        for left, right in pairwise(leaves):
            candidate = self.merge(left, right)
            heapq.heappush(merged, (candidate.gain, candidate))
        while merged:
            gain, node = heapq.heappop(merged)
            while node.left in removed or node.right in removed:
                if not merged:
                    break
                gain, node = heapq.heappop(merged)
            else:
                stop = True
                if n_bkps is not None:
                    if len(leaves) > n_bkps + 1:
                        stop = False
                elif pen is not None:
                    if gain < pen:
                        stop = False
                elif epsilon is not None:
                    if sum(leaf.val for leaf in leaves) < epsilon:
                        stop = False
                if stop:
                    break
                idx = bisect_left(keys, node.left.start)
                leaves[idx] = node
                keys[idx] = node.start
                del leaves[idx + 1]
                del keys[idx + 1]
                removed.add(node.left)
                removed.add(node.right)
                if idx > 0:
                    left_candidate = self.merge(leaves[idx - 1], node)
                    heapq.heappush(merged, (left_candidate.gain, left_candidate))
                if idx < len(leaves) - 1:
                    right_candidate = self.merge(node, leaves[idx + 1])
                    heapq.heappush(merged, (right_candidate.gain, right_candidate))
        return {(leaf.start, leaf.end): leaf.val for leaf in leaves}

    def fit(self, signal) -> "BottomUp":
        self.cost.fit(signal)
        if signal.ndim == 1:
            self.n_samples = signal.shape[0]
        else:
            self.n_samples = signal.shape[0]
        self.leaves = self._grow_tree()
        self.merge.cache_clear()
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

    def fit_predict(self, signal, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen, epsilon=epsilon)
