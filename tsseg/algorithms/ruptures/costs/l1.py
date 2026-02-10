"""Least absolute deviation cost."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


class CostL1(BaseCost):
    """Least absolute deviation cost."""

    model = "l1"

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.min_size = 2

    def fit(self, signal: np.ndarray) -> "CostL1":
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
        sub = self.signal[start:end]
        med = np.median(sub, axis=0)
        return float(np.abs(sub - med).sum())
