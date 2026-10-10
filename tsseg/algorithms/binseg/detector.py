"""BinSeg detector aligned with the tsseg BaseSegmenter API."""

from __future__ import annotations

import warnings
from typing import Any

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
from ..ruptures.base import BaseCost
from ..ruptures.detection.binseg import Binseg
from ..ruptures.utils import bic_penalty

__all__ = ["BinSegDetector"]


def _ensure_time_major(X: Any, *, axis: int) -> np.ndarray:
    """Coerce input into a 2D time-major array."""

    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif axis != 0:
        data = np.moveaxis(data, axis, 0)
    if data.ndim != 2:
        raise ValueError("BinSegDetector expects 1D or 2D inputs")
    return data


class BinSegDetector(BaseSegmenter):
    """Binary Segmentation change-point detector using ruptures' implementation.

    This wrapper fits the single-change ``Binseg`` solver on the provided data
    and repeatedly splits the series to obtain multiple change points.

    Parameters
    ----------
    n_cps : int or None, default=None
        Number of change points to return. When given, ``penalty`` and
        ``epsilon`` are ignored, with a ``UserWarning`` for each one that is not
        ``None`` (``penalty`` is 10 by default, so pass ``penalty=None`` to
        avoid the warning). When ``None``, splitting stops on ``penalty`` or
        ``epsilon``.
    model : str, default="l2"
        Cost model passed to ruptures. Examples include "l2", "l1", "rbf".
    min_size : int, default=2
        Minimum segment length enforced during the binary search.
    jump : int, default=5
        Sub-sampling factor for candidate breakpoints.
    penalty : float or None, default=10
        Penalty per change point, used when ``n_cps`` is ``None``: splitting
        continues while the best split lowers the cost by more than
        ``penalty``. Must be strictly positive. At most one of ``penalty`` and
        ``epsilon`` may be set, so ``epsilon`` requires ``penalty=None``.
    pen_scale : {None, "bic"}, default=None
        ``"bic"``: ``penalty`` is a coefficient on ``log(n) * u``, u the cost of
        one sample of the whole signal (d for ``"l2"`` on unit-variance
        channels, where it is the BIC penalty; 1 for the kernel costs).
    epsilon : float or None, default=None
        Reconstruction budget, used when ``n_cps`` is ``None`` and
        ``penalty=None``: splitting continues while the total cost of the
        segmentation exceeds ``epsilon``. Must be strictly positive.
    custom_cost : BaseCost, optional
        Pre-instantiated ruptures cost object.
    cost_params : dict, optional
        Additional keyword arguments forwarded to ``cost_factory`` when
        building the cost from ``model``.
    backend : {"auto", "numba", "python"}, default="auto"
        Backend of the costs ``l1``, ``l2``, ``rbf`` and ``cosine``: numba when
        it is installed under ``"auto"``, the kernel costs then without the
        n x n Gram matrix. Same change points either way.
    axis : int, default=0
        Axis representing time in the input array.

    Raises
    ------
    ValueError
        If ``n_cps``, ``penalty`` and ``epsilon`` are all ``None``, if both
        ``penalty`` and ``epsilon`` are set without ``n_cps``, or if
        ``penalty`` or ``epsilon`` is not strictly positive.
    """

    _tags = {
        "fit_is_empty": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "n_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points to detect.",
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
        "penalty": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Penalty threshold (mutually exclusive with n_cps/epsilon).",
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
            description="Reconstruction error tolerance.",
            nullable=True,
            group="stopping_criterion",
        ),
        "custom_cost": ParamDef(
            constraint=HasType((BaseCost,)),
            description="Pre-instantiated ruptures cost object.",
            nullable=True,
            ui_hidden=True,
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
            MutuallyExclusive(["n_cps", "penalty", "epsilon"], required_count=1),
            DataDependent(
                "n_cps < n_samples",
                "n_cps must be less than the number of samples",
            ),
            DataDependent(
                "min_size * 2 <= n_samples",
                "min_size is too large for the series length",
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
        custom_cost: BaseCost | None = None,
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
        self.custom_cost = custom_cost
        self.cost_params = cost_params or {}
        self.backend = backend
        self._estimator: Binseg | None = None
        self._fitted_change_points: np.ndarray | None = None
        super().__init__(axis=axis)

    def _make_estimator(self, signal: np.ndarray) -> Binseg:
        estimator = Binseg(
            model=self.model,
            custom_cost=self.custom_cost,
            min_size=self.min_size,
            jump=self.jump,
            params=self.cost_params or None,
            backend=self.backend,
        )
        estimator.fit(signal)
        return estimator

    def _fit(self, X, y=None):
        signal = _ensure_time_major(X, axis=self.axis)
        self._estimator = self._make_estimator(signal)
        self._fitted_change_points = None

        return self

    def _resolved_penalty(self) -> float | None:
        """``penalty``, or with ``pen_scale="bic"`` the penalty it scales
        (:func:`~tsseg.algorithms.ruptures.utils.bic_penalty`)."""
        if self.penalty is None or self.pen_scale is None:
            return self.penalty
        return bic_penalty(self.penalty, self._estimator.cost)

    def _predict(self, X, axis=None):
        axis = self.axis if axis is None else axis
        signal = _ensure_time_major(X, axis=axis)

        if self._estimator is None or not np.array_equal(
            signal, getattr(self._estimator, "signal", None)
        ):
            self._estimator = self._make_estimator(signal)

        bkps = self._estimator.predict(
            n_bkps=self.n_cps,
            pen=self._resolved_penalty(),
            epsilon=self.epsilon,
        )
        n_samples = self._estimator.n_samples or signal.shape[0]
        change_points = np.array([b for b in bkps if 0 < b < n_samples], dtype=int)
        change_points = np.unique(change_points)
        self._fitted_change_points = change_points.copy()
        return change_points

    def get_fitted_params(self):
        return {
            "n_cps": self.n_cps,
            "penalty": self.penalty,
            "epsilon": self.epsilon,
        }

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"n_cps": 1, "semi_supervised": True}
