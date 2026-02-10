"""Linear regression cost."""

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq

from ..base import BaseCost
from ..exceptions import NotEnoughPoints


class CostLinear(BaseCost):
    """Least squares regression residual cost."""

    model = "linear"

    def __init__(self) -> None:
        self.signal: np.ndarray | None = None
        self.covar: np.ndarray | None = None
        self.min_size = 2

    def fit(self, signal: np.ndarray) -> "CostLinear":
        if signal.ndim <= 1:
            raise ValueError("Linear cost expects at least two columns (target + covariates)")
        self.signal = signal[:, 0].reshape(-1, 1)
        self.covar = signal[:, 1:]
        return self

    def error(self, start: int, end: int) -> float:
        if self.signal is None or self.covar is None:
            raise RuntimeError("Cost not fitted")
        if end - start < self.min_size:
            raise NotEnoughPoints
        y = self.signal[start:end]
        X = self.covar[start:end]
        _, residuals, _, _ = lstsq(X, y, rcond=None)
        return float(residuals.sum() if residuals.size else 0.0)
