"""BinSeg detector aligned with the tsseg BaseSegmenter API."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from ..utils import extract_cps

from ..base import BaseSegmenter
from ..ruptures.detection.binseg import Binseg
from ..ruptures.base import BaseCost

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
        Number of change points to return. When ``None``, ruptures stops
        according to its internal criteria (``pen`` or ``epsilon``).
    model : str, default="l2"
        Cost model passed to ruptures. Examples include "l2", "l1", "rbf".
    min_size : int, default=2
        Minimum segment length enforced during the binary search.
    jump : int, default=5
        Sub-sampling factor for candidate breakpoints.
    penalty : float or None, default=None
        Penalty threshold supplied to ``predict``. Mutually exclusive with
        ``n_cps`` and ``epsilon``. Provide at least one of ``n_cps``, ``penalty``
        or ``epsilon``.
    epsilon : float or None, default=None
        Reconstruction error tolerance. Mutually exclusive with ``n_cps`` and
        ``penalty``.
    custom_cost : BaseCost, optional
        Pre-instantiated ruptures cost object.
    cost_params : dict, optional
        Additional keyword arguments forwarded to ``cost_factory`` when
        building the cost from ``model``.
    axis : int, default=0
        Axis representing time in the input array.

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

    def __init__(
        self,
        *,
        n_cps: int | None = None,
        model: str = "l2",
        min_size: int = 2,
        jump: int = 5,
        penalty: float | None = 10,
        epsilon: float | None = None,
        custom_cost: BaseCost | None = None,
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
        self.custom_cost = custom_cost
        self.cost_params = cost_params or {}
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
        )
        estimator.fit(signal)
        return estimator

    def _fit(self, X, y=None):
        signal = _ensure_time_major(X, axis=self.axis)
        self._estimator = self._make_estimator(signal)
        self._fitted_change_points = None

        return self

    def _predict(self, X, axis=None):
        axis = self.axis if axis is None else axis
        signal = _ensure_time_major(X, axis=axis)

        if self._estimator is None or not np.array_equal(signal, getattr(self._estimator, "signal", None)):
            self._estimator = self._make_estimator(signal)

        bkps = self._estimator.predict(
            n_bkps=self.n_cps,
            pen=self.penalty,
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