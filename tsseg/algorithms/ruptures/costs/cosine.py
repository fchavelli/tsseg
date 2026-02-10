"""Cosine similarity kernel cost."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


def _cosine_gram(X: np.ndarray) -> np.ndarray:
    dot = X @ X.T
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    denom = norms * norms.T
    with np.errstate(divide="ignore", invalid="ignore"):
        gram = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return gram


class CostCosine(BaseCost):
    """Kernel cost based on cosine similarity."""

    model = "cosine"

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self._gram: np.ndarray | None = None
        self.min_size = 1

    @property
    def gram(self) -> np.ndarray:
        if self.signal is None:
            raise RuntimeError("Cost not fitted")
        if self._gram is None:
            self._gram = _cosine_gram(self.signal)
        return self._gram

    def fit(self, signal: np.ndarray) -> "CostCosine":
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        self._gram = None
        return self

    def error(self, start: int, end: int) -> float:
        if end - start < self.min_size:
            raise NotEnoughPoints
        gram = self.gram
        sub = gram[start:end, start:end]
        return float(np.trace(sub) - sub.sum() / max(end - start, 1))
