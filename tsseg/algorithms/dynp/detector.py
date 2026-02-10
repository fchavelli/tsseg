"""Dynamic programming detector built on the vendored ruptures implementation."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter
from .. import utils
from ..ruptures.detection import Dynp

__all__ = ["DynpDetector"]


class DynpDetector(BaseSegmenter):
    """Exact dynamic programming change point detector."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        *,
        n_cps: int = 1,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        cost_params: dict | None = None,
        semi_supervised: bool = True,
        axis: int = 0,
    ) -> None:
        if n_cps is None:
            raise ValueError("n_cps must be a positive integer (required by the DP solver).")
        if int(n_cps) <= 0:
            raise ValueError("n_cps must be a positive integer.")
        self.n_cps = int(n_cps)
        self.model = model
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.cost_params = cost_params or {}
        self._estimator: Dynp | None = None
        self._train_signal: np.ndarray | None = None
        self.semi_supervised: bool = semi_supervised
        super().__init__(axis=axis)
        
        # if self.n_cps is not None:
        #     self._tags_dynamic["capability:unsupervised"] = True

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError("DynpDetector expects 1D or 2D arrays")
        return X

    def _fit(self, X, y):
        signal = self._ensure_2d(X)
        estimator = Dynp(
            model=self.model,
            min_size=self.min_size,
            jump=self.jump,
            params=self.cost_params,
        )
        estimator.fit(signal)
        self._estimator = estimator
        self._train_signal = signal
        return self

    def _predict(self, X):
        if self._estimator is None:
            raise RuntimeError("DynpDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = np.asarray(self._estimator.predict(n_bkps=self.n_cps), dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return np.unique(bkps)
