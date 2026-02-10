"""Radial basis function kernel cost."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


def _pairwise_sq_euclidean(X: np.ndarray) -> np.ndarray:
    norms = np.sum(X * X, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    return dist2


class CostRbf(BaseCost):
    """Kernel cost using an RBF kernel."""

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
            dist2 = _pairwise_sq_euclidean(self.signal)
            gamma = self.gamma
            if gamma is None:
                median = np.median(dist2[dist2 > 0]) if np.any(dist2 > 0) else 1.0
                gamma = 1.0 / median if median != 0 else 1.0
                self.gamma = gamma
            self._gram = np.exp(-gamma * dist2)
        return self._gram

    def fit(self, signal: np.ndarray) -> "CostRbf":
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self._gram = None
        if self.gamma is None:
            _ = self.gram
        return self

    def error(self, start: int, end: int) -> float:
        if end - start < self.min_size:
            raise NotEnoughPoints
        gram = self.gram
        sub = gram[start:end, start:end]
        val = float(np.trace(sub) - sub.sum() / max(end - start, 1))
        return val
