"""Bottom-up segmentation algorithm."""

from __future__ import annotations

import heapq
from bisect import bisect_left
from functools import lru_cache

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import Bnode, pairwise, sanity_check, tie_tol, tie_unit


class BottomUp(BaseEstimator):
    """Bottom-up agglomerative segmentation.

    ``backend``: ``"auto"`` computes the costs ``l1``, ``l2``, ``rbf`` and
    ``cosine`` with numba (``ruptures.accel``) when it is installed, the kernel
    costs without the n x n Gram matrix; ``"numba"`` and ``"python"`` force
    either path. Both return the same change points.

    Merge gains closer than the tie gap of the merged segments' costs
    (``utils.tie_tol``) are equal, and the leftmost of the tied merges is done
    first, as the heap orders exact ties by ``Bnode.start``; a gain tied with
    ``pen`` stops.
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
            cost = self._error(start, end)
            leaves.append(Bnode(start, end, cost))
        return leaves

    @lru_cache(maxsize=None)
    def merge(self, left: Bnode, right: Bnode) -> Bnode:
        if left.end != right.start:
            raise ValueError("Segments must be contiguous")
        start, end = left.start, right.end
        val = self._error(start, end)
        self._top_val = max(self._top_val, abs(val))
        return Bnode(start, end, val, left=left, right=right)

    def _tol(self, *nodes: Bnode) -> float:
        """Tie gap of merge gains computed from these merged segments."""
        length = max(node.end - node.start for node in nodes)
        return tie_tol(*(node.val for node in nodes), length * self._unit)

    def _pop(self, merged, removed) -> Bnode | None:
        """Pop the cheapest merge whose halves are still leaves. The merges
        tied with it are popped too and the leftmost one is returned; the
        others go back on the heap."""
        while merged:
            gain, node = heapq.heappop(merged)
            if node.left not in removed and node.right not in removed:
                break
        else:
            return None
        # no merge beyond this gain can tie: all segments cost at most _top_val
        bound = gain + tie_tol(self._top_val, self.n_samples * self._unit)
        tied = [(gain, node)]
        while merged and merged[0][0] <= bound:
            other_gain, other = heapq.heappop(merged)
            if other.left not in removed and other.right not in removed:
                tied.append((other_gain, other))
        chosen = min(
            (other for other_gain, other in tied if other_gain <= gain + self._tol(node, other)),
            key=lambda other: other.start,
        )
        for item in tied:
            if item[1] is not chosen:
                heapq.heappush(merged, item)
        return chosen

    def _seg(self, n_bkps: int | None = None, pen: float | None = None, epsilon: float | None = None):
        leaves = sorted(self.leaves)
        keys = [leaf.start for leaf in leaves]
        removed: set[Bnode] = set()
        merged: list[tuple[float, Bnode]] = []
        for left, right in pairwise(leaves):
            candidate = self.merge(left, right)
            heapq.heappush(merged, (candidate.gain, candidate))
        while merged:
            node = self._pop(merged, removed)
            if node is None:
                break
            gain = node.gain
            stop = True
            if n_bkps is not None:
                if len(leaves) > n_bkps + 1:
                    stop = False
            elif pen is not None:
                if gain < pen - self._tol(node):
                    stop = False
            elif epsilon is not None:
                error = sum(leaf.val for leaf in leaves)
                if error < epsilon - tie_tol(error, self.n_samples * self._unit):
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
        from .. import accel  # numba, imported on first use

        self.cost.fit(signal)
        if signal.ndim == 1:
            self.n_samples = signal.shape[0]
        else:
            self.n_samples = signal.shape[0]
        self._unit = tie_unit(self.cost)
        if accel.use_numba(self.cost, self.backend):
            self._error = accel.Signal(self.cost).error
        else:
            self._error = self.cost.error
        self.leaves = self._grow_tree()
        self._top_val = max(abs(leaf.val) for leaf in self.leaves)
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
