"""Radial basis function kernel cost."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


def median_sq_dist(signal: np.ndarray) -> float:
    """Median of the positive pairwise squared distances, 0.0 if there are none.

    In O(n) memory with numba; otherwise over ``pdist``, O(n^2).
    """
    from .. import ekcpd

    if ekcpd.AVAILABLE:
        return ekcpd.median_sq_dist(signal)
    dist2 = pdist(signal, "sqeuclidean")
    dist2 = dist2[dist2 > 0]
    return float(np.median(dist2)) if dist2.size else 0.0


class CostRbf(BaseCost):
    """Kernel cost using an RBF kernel.

    ``gamma=None`` applies the median heuristic: 1 / median of the positive
    squared distances between distinct samples. It is set by :meth:`fit`
    without the Gram matrix, which only :meth:`error` builds.
    """

    model = "rbf"

    def __init__(self, gamma: float | None = None) -> None:
        self.gamma = gamma
        self.signal: np.ndarray | None = None
        self._gram: np.ndarray | None = None
        self.min_size = 1

    @property
    def gram(self) -> np.ndarray:
        if self.signal is None:
            raise RuntimeError("Cost not fitted")
        if self._gram is None:
            if self.gamma is None:
                self._set_gamma()
            dist2 = squareform(pdist(self.signal, "sqeuclidean"))
            self._gram = np.exp(-self.gamma * dist2)
        return self._gram

    def _set_gamma(self) -> None:
        median = median_sq_dist(self.signal)
        self.gamma = 1.0 / median if median != 0 else 1.0

    def fit(self, signal: np.ndarray) -> "CostRbf":
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self._gram = None
        if self.gamma is None:
            self._set_gamma()
        return self

    def error(self, start: int, end: int) -> float:
        if end - start < self.min_size:
            raise NotEnoughPoints
        gram = self.gram
        sub = gram[start:end, start:end]
        val = float(np.trace(sub) - sub.sum() / max(end - start, 1))
        return val
