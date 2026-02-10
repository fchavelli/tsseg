"""Least-squares cost."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


class CostL2(BaseCost):
    """Least squared deviation cost."""

    model = "l2"

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.min_size = 1

    def fit(self, signal: np.ndarray) -> "CostL2":
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
        return float(segment.var(axis=0, ddof=0).sum() * (end - start))
