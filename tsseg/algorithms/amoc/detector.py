"""AMOC detector integrated with aeon's BaseSegmenter API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..base import BaseSegmenter

__all__ = ["AmocDetector"]


def _ensure_time_major(X, *, axis: int | None) -> np.ndarray:
    """Convert input array to time-major shape (n_timepoints, n_features)."""

    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif axis is not None and axis != 0:
        data = np.moveaxis(data, axis, 0)
    if data.ndim != 2:
        raise ValueError("Input time series must be one- or two-dimensional")
    return data


@dataclass
class _AmocEngine:
    """Internal helper implementing the AMOC objective."""

    min_size: int = 1
    signal: Optional[np.ndarray] = None

    def fit(self, signal: np.ndarray) -> "_AmocEngine":
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]
        if signal.ndim != 2:
            raise ValueError("Signal must be one- or two-dimensional")
        self.signal = signal
        return self

    def predict(self) -> np.ndarray:
        if self.signal is None:
            raise RuntimeError("AMOC engine must be fitted before prediction")

        n_timepoints = self.signal.shape[0]
        if n_timepoints < 2 * self.min_size:
            return np.array([], dtype=int)

        sse = np.full(n_timepoints - 1, np.inf, dtype=float)
        for t in range(self.min_size, n_timepoints - self.min_size + 1):
            left = self.signal[:t]
            right = self.signal[t:]

            mean_left = left.mean(axis=0, keepdims=True)
            mean_right = right.mean(axis=0, keepdims=True)

            error_left = np.sum((left - mean_left) ** 2)
            error_right = np.sum((right - mean_right) ** 2)

            sse[t - 1] = error_left + error_right

        if not np.isfinite(sse).any():
            return np.array([], dtype=int)

        cp_idx = int(np.argmin(sse)) + 1
        return np.array([cp_idx], dtype=int)


class AmocDetector(BaseSegmenter):
    """AMOC (At Most One Change) detector.

    The AMOC objective searches for the breakpoint that minimises the sum of
    squared errors on either side of the split. It is a foundational building
    block for many multi-change detectors (e.g. Binary Segmentation and PELT),
    which repeatedly apply the single-change solver on sub-segments of the
    signal.

    Parameters
    ----------
    min_size : int, default = 5
        Minimum number of samples required on each side of the breakpoint.
    axis : int, default = 0
        Axis representing time in the input array.

    References
    ----------
    .. [1] https://cran.r-project.org/web/packages/changepoint/
    .. [2] https://github.com/rkillick/changepoint/

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms.amoc.detector import AmocDetector
    >>> rng = np.random.default_rng(7)
    >>> x = np.concatenate([
    ...     rng.normal(0.0, 0.2, size=200),
    ...     rng.normal(1.0, 0.2, size=200),
    ... ])
    >>> detector = AmocDetector(min_size=20)
    >>> detector.fit(x)
    AmocDetector(...)
    >>> detector.predict(x)  # doctest: +SKIP
    array([...])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "semi_supervised": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    def __init__(self, *, min_size: int = 5, axis: int = 0) -> None:
        if min_size < 1:
            raise ValueError("min_size must be at least 1")
        self.min_size = int(min_size)
        self._engine: _AmocEngine | None = None
        self._fitted_change_points: np.ndarray | None = None
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        signal = _ensure_time_major(X, axis=self.axis)
        self._engine = _AmocEngine(min_size=self.min_size).fit(signal)
        self._fitted_change_points = None
        return self

    def _predict(self, X, axis=None):
        axis = self.axis if axis is None else axis
        signal = _ensure_time_major(X, axis=axis)

        if self._engine is None or self._engine.signal is None:
            self._engine = _AmocEngine(min_size=self.min_size).fit(signal)
        elif not np.array_equal(signal, self._engine.signal):
            self._engine = _AmocEngine(min_size=self.min_size).fit(signal)

        change_points = self._engine.predict()
        valid = (change_points > 0) & (change_points < signal.shape[0])
        change_points = change_points[valid]
        self._fitted_change_points = change_points.copy()
        return change_points

    def get_fitted_params(self):
        if self._fitted_change_points is None:
            return {"change_points": None}
        return {"change_points": self._fitted_change_points.copy()}