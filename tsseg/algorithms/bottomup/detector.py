"""Bottom-up detector built on the vendored ruptures implementation."""

from __future__ import annotations

import warnings

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    Interval,
    MutuallyExclusive,
    ParamDef,
    StrOptions,
)
from ..ruptures.detection import BottomUp
from ..ruptures.utils import bic_penalty

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

    _parameter_schema = {
        "n_cps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of change points to detect.",
            nullable=True,
            group="stopping_criterion",
        ),
        "model": ParamDef(
            constraint=StrOptions({"l1", "l2", "rbf", "linear", "normal", "cosine"}),
            description="Cost model for segment homogeneity.",
        ),
        "min_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum segment length.",
        ),
        "jump": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Grid step for candidate change points.",
        ),
        "penalty": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Penalty value (must be > 0).",
            nullable=True,
            group="stopping_criterion",
        ),
        "pen_scale": ParamDef(
            constraint=StrOptions({"bic"}),
            description=(
                "How ``penalty`` is scaled. ``None``: it is the penalty. ``'bic'``: "
                "a coefficient on ``log(n) * u``, u the cost of one sample of the "
                "whole signal (d for l2 on unit-variance channels, where it is the "
                "BIC penalty; 1 for the kernel costs), so that one value applies "
                "across lengths, dimensions and costs."
            ),
            nullable=True,
            group="stopping_criterion",
        ),
        "epsilon": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Reconstruction budget (must be > 0).",
            nullable=True,
            group="stopping_criterion",
        ),
        "cost_params": ParamDef(
            constraint=None,
            description="Extra params passed to the cost function.",
            nullable=True,
            ui_hidden=True,
        ),
        "backend": ParamDef(
            constraint=StrOptions({"auto", "numba", "python"}),
            description=(
                "Backend of the costs l1, l2, rbf and cosine. ``auto``: numba "
                "when it is installed (the kernel costs then in O(n) memory, "
                "without the n x n Gram matrix), else Python. Same segmentation "
                "either way."
            ),
            ui_hidden=True,
        ),
        "_cross_constraints": [
            MutuallyExclusive(
                ["n_cps", "penalty", "epsilon"],
                required_count=1,
            ),
            DataDependent(
                "min_size * 2 <= n_samples",
                "Series must have at least 2 × min_size samples.",
            ),
        ],
    }

    def __init__(
        self,
        *,
        n_cps: int | None = None,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        penalty: float | None = 10,
        pen_scale: str | None = None,
        epsilon: float | None = None,
        cost_params: dict | None = None,
        backend: str = "auto",
        axis: int = 0,
    ) -> None:
        if pen_scale not in (None, "bic"):
            raise ValueError(f"pen_scale must be None or 'bic', got {pen_scale!r}")
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
        self.pen_scale = pen_scale
        self.epsilon = epsilon_value
        self.cost_params = cost_params or {}
        self.backend = backend
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
            backend=self.backend,
        )
        estimator.fit(signal)
        self._estimator = estimator
        self._train_signal = signal

        return self

    def _resolved_penalty(self) -> float | None:
        """``penalty``, or with ``pen_scale="bic"`` the penalty it scales
        (:func:`~tsseg.algorithms.ruptures.utils.bic_penalty`)."""
        if self.penalty is None or self.pen_scale is None:
            return self.penalty
        return bic_penalty(self.penalty, self._estimator.cost)

    def _predict(self, X):
        if self._estimator is None:
            raise RuntimeError("BottomUpDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = np.asarray(
            self._estimator.predict(
                n_bkps=self.n_cps, pen=self._resolved_penalty(), epsilon=self.epsilon
            ),
            dtype=int,
        )
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return np.unique(bkps)
