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
from ..ruptures.utils import bic_penalty

__all__ = ["WindowDetector"]


class WindowDetector(BaseSegmenter):
    """Sliding-window change point detector using the vendored ruptures implementation.

    A window of ``width`` samples slides over the series; at each candidate
    point, the score is the cost of the whole window minus the costs of its
    two halves. The local maxima of this score are then added as change
    points, highest score first, until the stopping criterion is met.

    Parameters
    ----------
    width : int, default=100
        Window length, rounded down to an even number. Candidates within
        ``width // 2`` samples of either end of the series are not scored.
    n_cps : int or None, default=None
        Number of change points to return; fewer are returned when the score
        has fewer local maxima. Takes precedence over ``pen`` and ``epsilon``
        (with a ``UserWarning`` if either is also set).
    pen : float or None, default=None
        Penalty per change point: a peak is added while it lowers the cost of
        the segmentation of the whole series by more than ``pen``. Takes
        precedence over ``epsilon`` (with a ``UserWarning`` if both are set).
    pen_scale : {None, "bic"}, default=None
        ``None``: ``pen`` is used as is. ``"bic"``: ``pen`` is a coefficient on
        ``log(n) * u``, u the cost of one sample of the whole signal (d for
        ``"l2"`` on unit-variance channels, where it is the BIC penalty; 1 for
        the kernel costs).
    epsilon : float or None, default=None
        Reconstruction budget: peaks are added while the total cost of the
        segmentation exceeds ``epsilon``.
    model : {"l2", "l1", "rbf", "linear", "normal", "cosine"}, default="l2"
        Cost model: least squares (``"l2"``), least absolute deviation
        (``"l1"``), Gaussian kernel (``"rbf"``), linear regression
        (``"linear"``), Gaussian likelihood (``"normal"``) or cosine kernel
        (``"cosine"``).
    min_size : int, default=2
        Minimum segment length. Must not exceed half the series length.
    jump : int, default=5
        Sub-sampling factor for candidate breakpoints: only multiples of
        ``jump`` are scored.
    cost_params : dict or None, default=None
        Extra keyword arguments forwarded to the cost (for instance
        ``{"gamma": 0.1}`` for ``"rbf"``).
    backend : {"auto", "numba", "python"}, default="auto"
        Backend of the costs ``l1``, ``l2``, ``rbf`` and ``cosine``: numba when
        it is installed under ``"auto"``, the kernel costs then without the
        n x n Gram matrix. Same change points either way.
    axis : int, default=0
        Axis of ``X`` that represents time.

    Raises
    ------
    ValueError
        If ``n_cps``, ``pen`` and ``epsilon`` are all ``None``, which is the
        default: one stopping criterion must be given.
    """

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
            constraint=Interval(int, 0, None, Closed.LEFT),
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
        "pen_scale": ParamDef(
            constraint=StrOptions({"bic"}),
            description=(
                "How ``pen`` is scaled. ``None``: it is the penalty. ``'bic'``: "
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
        pen_scale: str | None = None,
        epsilon: float | None = None,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        cost_params: dict | None = None,
        backend: str = "auto",
        axis: int = 0,
    ) -> None:
        if pen_scale not in (None, "bic"):
            raise ValueError(f"pen_scale must be None or 'bic', got {pen_scale!r}")
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
        self.pen_scale = pen_scale
        self.epsilon = None if epsilon_value is None else float(epsilon_value)
        self.model = model
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.cost_params = cost_params or {}
        self.backend = backend
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
            backend=self.backend,
        )
        estimator.fit(signal)
        self._estimator = estimator
        self._train_signal = signal

        return self

    def _resolved_penalty(self) -> float | None:
        """``pen``, or with ``pen_scale="bic"`` the penalty it scales
        (:func:`~tsseg.algorithms.ruptures.utils.bic_penalty`)."""
        if self.pen is None or self.pen_scale is None:
            return self.pen
        return bic_penalty(self.pen, self._estimator.cost)

    def _predict(self, X):
        if self._estimator is None:
            raise RuntimeError("WindowDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = self._estimator.predict(
            n_bkps=self.n_cps,
            pen=self._resolved_penalty(),
            epsilon=self.epsilon,
        )
        bkps = np.asarray(bkps, dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return np.unique(bkps)
