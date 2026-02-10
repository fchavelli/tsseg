"""Kernel-based change point detection using existing dynamic programming tools."""

from __future__ import annotations

import numpy as np

from ..base import BaseCost, BaseEstimator
from ..costs import cost_factory
from ..exceptions import BadSegmentationParameters
from ..utils import sanity_check
from .dynp import Dynp
from .pelt import Pelt


class KernelCPD(BaseEstimator):
    """Kernelised change point detection with DP or PELT backends."""

    def __init__(
        self,
        kernel: str = "linear",
        custom_cost: BaseCost | None = None,
        min_size: int = 2,
        jump: int = 1,
        params: dict | None = None,
    ) -> None:
        if kernel not in {"linear", "rbf", "cosine"}:
            raise ValueError(f"Unsupported kernel '{kernel}'")
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
            self.params = params or {}
        else:
            kwargs = params or {}
            model_name = "l2" if kernel == "linear" else kernel
            self.cost = cost_factory(model=model_name, **kwargs)
            self.params = kwargs
        self.kernel = kernel
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = max(jump, 1)
        self.n_samples: int | None = None
        self.signal: np.ndarray | None = None

    def fit(self, signal) -> "KernelCPD":
        array = np.asarray(signal, dtype=float)
        if array.ndim == 1:
            array = array[:, np.newaxis]
        self.cost.fit(array)
        self.signal = array
        self.n_samples = array.shape[0]
        return self

    def predict(self, n_bkps: int | None = None, pen: float | None = None):
        if self.signal is None:
            raise RuntimeError("KernelCPD must be fitted before predict")
        if n_bkps is not None and pen is not None:
            raise ValueError("Provide either n_bkps or pen, not both")
        if n_bkps is None and pen is None:
            raise ValueError("Provide a stopping criterion (n_bkps or pen)")
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters
        if n_bkps is not None:
            estimator = Dynp(
                custom_cost=self.cost,
                min_size=self.min_size,
                jump=self.jump,
            )
            estimator.fit(self.signal)
            return estimator.predict(n_bkps=n_bkps)
        pen = float(pen)
        if pen <= 0:
            raise ValueError("pen must be positive")
        estimator = Pelt(
            custom_cost=self.cost,
            min_size=self.min_size,
            jump=self.jump,
        )
        estimator.fit(self.signal)
        return estimator.predict(pen=pen)

    def fit_predict(self, signal, n_bkps: int | None = None, pen: float | None = None):
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen)
