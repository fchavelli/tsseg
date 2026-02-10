"""Gaussian likelihood cost."""

from __future__ import annotations

import numpy as np
from numpy.linalg import slogdet

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


class CostNormal(BaseCost):
    """Cost based on the Gaussian log-likelihood."""

    model = "normal"

    def __init__(self, add_small_diag: bool = True) -> None:
        self.signal: np.ndarray | None = None
        self.add_small_diag = add_small_diag
        self.min_size = 2

    def fit(self, signal: np.ndarray) -> "CostNormal":
        if signal.ndim == 1:
            self.signal = signal.reshape(-1, 1)
        else:
            self.signal = signal
        return self

    def error(self, start: int, end: int) -> float:
        if self.signal is None:
            raise RuntimeError("Cost not fitted")
        if end - start < self.min_size:
            raise NotEnoughPoints
        segment = self.signal[start:end]
        if segment.shape[1] > 1:
            cov = np.cov(segment.T)
        else:
            cov = np.array([[segment.var(ddof=0)]])
        if self.add_small_diag:
            cov += 1e-6 * np.eye(cov.shape[0])
        sign, val = slogdet(cov)
        if sign <= 0:
            return float("inf")
        return float(val * (end - start))
