"""Bottom-up detector built on the vendored ruptures implementation."""

from __future__ import annotations

import warnings

import numpy as np

from ..base import BaseSegmenter
from ..ruptures.detection import BottomUp

__all__ = ["BottomUpDetector"]


class BottomUpDetector(BaseSegmenter):
    """Bottom-up change point detector."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        *,
        n_cps: int | None = None,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        penalty: float | None = 10,
        epsilon: float | None = None,
        cost_params: dict | None = None,
        axis: int = 0,
    ) -> None:
        self.n_cps = None if n_cps is None else int(n_cps)
        penalty_value = penalty
        epsilon_value = epsilon

        if self.n_cps is None:
            if penalty_value is None and epsilon_value is None:
                raise ValueError("Provide at least one of n_cps, penalty, epsilon")
            if penalty_value is not None and epsilon_value is not None:
                raise ValueError("Provide at most one of n_cps, penalty, epsilon")
        else:
            if penalty_value is not None:
                warnings.warn(
                    "penalty is ignored when n_cps is provided; proceeding with n_cps only",
                    UserWarning,
                )
                penalty_value = None
            if epsilon_value is not None:
                warnings.warn(
                    "epsilon is ignored when n_cps is provided; proceeding with n_cps only",
                    UserWarning,
                )
                epsilon_value = None

        if penalty_value is not None:
            if penalty_value <= 0:
                raise ValueError("penalty must be strictly positive")
            penalty_value = float(penalty_value)
        if epsilon_value is not None:
            if epsilon_value <= 0:
                raise ValueError("epsilon must be strictly positive")
            epsilon_value = float(epsilon_value)

        self.model = model
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.penalty = penalty_value
        self.epsilon = epsilon_value
        self.cost_params = cost_params or {}
        self._estimator: BottomUp | None = None
        self._train_signal: np.ndarray | None = None
        super().__init__(axis=axis)

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X[:, np.newaxis]
        if X.ndim != 2:
            raise ValueError("BottomUpDetector expects 1D or 2D arrays")
        return X

    def _fit(self, X, y=None):
        signal = self._ensure_2d(X)
        estimator = BottomUp(
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
            raise RuntimeError("BottomUpDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = np.asarray(self._estimator.predict(n_bkps=self.n_cps, pen=self.penalty, epsilon=self.epsilon), dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return np.unique(bkps)
