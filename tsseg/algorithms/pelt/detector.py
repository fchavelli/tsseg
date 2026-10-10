"""PELT change point detector using the vendored ruptures implementation."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)
from ..ruptures.detection import Pelt
from ..ruptures.utils import bic_penalty

__all__ = ["PeltDetector"]


class PeltDetector(BaseSegmenter):
    """Wrapper around the vendored ruptures :class:`Pelt` estimator.

    PELT minimises the sum of the segment costs plus ``penalty`` per change
    point. The number of change points follows from the penalty; it cannot be
    set directly (use :class:`~tsseg.algorithms.DynpDetector` for that).

    Parameters
    ----------
    model : {"l2", "l1", "rbf", "linear", "normal", "cosine"}, default="l2"
        Cost model: least squares (``"l2"``), least absolute deviation
        (``"l1"``), Gaussian kernel (``"rbf"``), linear regression
        (``"linear"``), Gaussian likelihood (``"normal"``) or cosine kernel
        (``"cosine"``).
    min_size : int, default=2
        Minimum segment length. Must not exceed half the series length.
    jump : int, default=5
        Sub-sampling factor for candidate breakpoints: only multiples of
        ``jump`` are considered.
    penalty : float, default=10.0
        Penalty added per change point. Must be strictly positive.
    pen_scale : {None, "bic"}, default=None
        ``None``: ``penalty`` is used as is. ``"bic"``: ``penalty`` is a
        coefficient on ``log(n) * u``, u the cost of one sample of the whole
        signal (d for ``"l2"`` on unit-variance channels, where it is the BIC
        penalty; 1 for the kernel costs).
    cost_params : dict or None, default=None
        Extra keyword arguments forwarded to the cost (for instance
        ``{"gamma": 0.1}`` for ``"rbf"``).
    backend : {"auto", "numba", "python"}, default="auto"
        Backend of the costs ``l1``, ``l2``, ``rbf`` and ``cosine``: numba when
        it is installed under ``"auto"``, the kernel costs then without the
        n x n Gram matrix. Same change points either way.
    axis : int, default=0
        Axis of ``X`` that represents time.

    Notes
    -----
    For a univariate signal with ``model="l1"``, upstream ruptures now ships
    ``L1Potts``, an exact and much faster solver of the same penalised problem
    with ``min_size=1`` and ``jump=1``; it is not vendored here. See
    https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/l1potts.py
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    _parameter_schema = {
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
        "penalty": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Penalty value for the PELT stopping criterion.",
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
        "cost_params": ParamDef(
            constraint=HasType((dict,)),
            description="Extra kwargs for cost_factory.",
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
            DataDependent(
                "min_size * 2 <= n_samples",
                "min_size is too large for the series length",
            ),
        ],
    }

    def __init__(
        self,
        *,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        penalty: float = 10.0,
        pen_scale: str | None = None,
        cost_params: dict | None = None,
        backend: str = "auto",
        axis: int = 0,
    ) -> None:
        if pen_scale not in (None, "bic"):
            raise ValueError(f"pen_scale must be None or 'bic', got {pen_scale!r}")
        self.penalty = float(penalty)
        self.pen_scale = pen_scale
        self.model = model
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.cost_params = cost_params or {}
        self.backend = backend
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
            backend=self.backend,
        )
        estimator.fit(signal)
        self._estimator = estimator
        self._train_signal = signal
        self._change_points = None
        return self

    def _resolved_penalty(self) -> float | None:
        """``penalty``, or with ``pen_scale="bic"`` the penalty it scales
        (:func:`~tsseg.algorithms.ruptures.utils.bic_penalty`)."""
        if self.penalty is None or self.pen_scale is None:
            return self.penalty
        return bic_penalty(self.penalty, self._estimator.cost)

    def _predict(self, X):
        if self._estimator is None:
            raise RuntimeError("PeltDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = np.asarray(self._estimator.predict(self._resolved_penalty()), dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        bkps = np.unique(bkps)
        self._change_points = bkps
        return bkps

    @property
    def change_points_(self) -> np.ndarray:
        """Change points returned by the last call to ``predict`` (a copy)."""
        if self._change_points is None:
            raise RuntimeError("Predict must be called before accessing change_points_")
        return self._change_points.copy()
