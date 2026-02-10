"""PELT change point detector using the vendored ruptures implementation."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter
from ..ruptures.detection import Pelt

__all__ = ["PeltDetector"]


class PeltDetector(BaseSegmenter):
    """Wrapper around the vendored ruptures :class:`Pelt` estimator."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    def __init__(
        self,
        *,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        penalty: float = 10.0,
        cost_params: dict | None = None,
        axis: int = 0,
    ) -> None:
        self.penalty = float(penalty)
        self.model = model
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.cost_params = cost_params or {}
        self._estimator: Pelt | None = None
        self._train_signal: np.ndarray | None = None
        self._change_points: np.ndarray | None = None
        super().__init__(axis=axis)

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError("PeltDetector expects 1D or 2D arrays")
        return X

    def _fit(self, X, y=None):
        signal = self._ensure_2d(X)
        estimator = Pelt(
            model=self.model,
            min_size=self.min_size,
            jump=self.jump,
            params=self.cost_params,
        )
        estimator.fit(signal)
        self._estimator = estimator
        self._train_signal = signal
        self._change_points = None
        return self

    def _predict(self, X):
        if self._estimator is None:
            raise RuntimeError("PeltDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = np.asarray(self._estimator.predict(self.penalty), dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        bkps = np.unique(bkps)
        self._change_points = bkps
        return bkps

    @property
    def change_points_(self) -> np.ndarray:
        if self._change_points is None:
            raise RuntimeError("Predict must be called before accessing change_points_")
        return self._change_points.copy()