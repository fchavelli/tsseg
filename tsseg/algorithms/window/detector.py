"""Window-based change point detector using the vendored ruptures implementation."""

from __future__ import annotations

import warnings

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    MutuallyExclusive,
    ParamDef,
    StrOptions,
)
from ..ruptures.detection import Window

__all__ = ["WindowDetector"]


class WindowDetector(BaseSegmenter):
    """Sliding window detector leveraging gain-based scoring."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "python_dependencies": None,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "width": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Width of the sliding window.",
            group="windowing",
        ),
        "n_cps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of change points to detect.",
            nullable=True,
            group="stopping_criterion",
        ),
        "pen": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Penalty threshold.",
            nullable=True,
            group="stopping_criterion",
        ),
        "epsilon": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Reconstruction error tolerance.",
            nullable=True,
            group="stopping_criterion",
        ),
        "model": ParamDef(
            constraint=StrOptions({"l1", "l2", "rbf", "linear", "normal", "cosine"}),
            description="Ruptures cost model name.",
        ),
        "min_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum segment length.",
        ),
        "jump": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Sub-sampling factor for candidate breakpoints.",
        ),
        "cost_params": ParamDef(
            constraint=HasType((dict,)),
            description="Extra kwargs for the cost function.",
            nullable=True,
            ui_hidden=True,
        ),
        "_cross_constraints": [
            MutuallyExclusive(["n_cps", "pen", "epsilon"], required_count=1),
            DataDependent(
                "width <= n_samples",
                "Window width must be <= series length",
            ),
        ],
    }

    def __init__(
        self,
        *,
        width: int = 100,
        n_cps: int | None = None,
        pen: float | None = None,
        epsilon: float | None = None,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        cost_params: dict | None = None,
        axis: int = 0,
    ) -> None:
        criteria = [n_cps is not None, pen is not None, epsilon is not None]
        if not any(criteria):
            raise ValueError("Configure at least one stopping criterion")

        penalty_value = pen
        epsilon_value = epsilon

        if sum(criteria) > 1:
            if n_cps is not None:
                warnings.warn(
                    "n_cps is provided together with pen/epsilon; ignoring pen and epsilon in favour of n_cps",
                    UserWarning,
                )
                penalty_value = None
                epsilon_value = None
            elif pen is not None and epsilon is not None:
                warnings.warn(
                    "pen and epsilon provided together; ignoring epsilon and using pen",
                    UserWarning,
                )
                epsilon_value = None

        self.width = int(width)
        self.n_cps = None if n_cps is None else int(n_cps)
        self.pen = None if penalty_value is None else float(penalty_value)
        self.epsilon = None if epsilon_value is None else float(epsilon_value)
        self.model = model
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.cost_params = cost_params or {}
        self._estimator: Window | None = None
        self._train_signal: np.ndarray | None = None
        super().__init__(axis=axis)

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            return array[:, np.newaxis]
        if array.ndim != 2:
            raise ValueError("WindowDetector expects 1D or 2D arrays")
        return array

    def _fit(self, X, y=None):
        signal = self._ensure_2d(X)
        estimator = Window(
            width=self.width,
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
            raise RuntimeError("WindowDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = self._estimator.predict(
            n_bkps=self.n_cps,
            pen=self.pen,
            epsilon=self.epsilon,
        )
        bkps = np.asarray(bkps, dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return np.unique(bkps)
