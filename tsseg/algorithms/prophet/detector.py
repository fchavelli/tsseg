"""Prophet-based change-point detector exposed as an aeon segmenter."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseSegmenter
from ..utils import multivariate_l2_norm, aggregate_change_points

from prophet import Prophet
from typing import Callable
from collections import Counter

# pandas Timestamp (ns resolution) overflows around 2262-04-11.
# From start="2000-01-01" that leaves ~95 700 days with freq="D".
# For longer series we fall back to freq="s" to stay within bounds.
_MAX_DAILY_PERIODS = 90_000  # conservative safety margin


def _safe_date_range(n_periods: int) -> pd.DatetimeIndex:
    """Build a DatetimeIndex that fits within pandas Timestamp limits.

    Uses daily frequency when possible (preserves Prophet's automatic
    weekly / yearly seasonality), and falls back to seconds for very
    long series that would overflow the nanosecond Timestamp range.
    """
    freq = "D" if n_periods <= _MAX_DAILY_PERIODS else "s"
    return pd.date_range(start="2000-01-01", periods=n_periods, freq=freq)

class ProphetDetector(BaseSegmenter):
    """Prophet forecaster wrapped as an aeon-compatible change-point detector."""

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
        n_changepoints: int | None = 5,
        n_changepoint_func: Callable[[np.ndarray], int] | None = None,
        axis: int = 0,
        multivariate_strategy: str = "ensembling",
        tolerance: int | float = 0.01,
        #n_segments: int | None = None,
    ) -> None:
        super().__init__(axis=axis)
        self.n_changepoints = int(n_changepoints) if n_changepoints is not None else None
        self.n_changepoint_func = n_changepoint_func
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        self.changepoints_: np.ndarray = np.empty(0, dtype=int)
        self.changepoint_strengths_: np.ndarray = np.empty(0, dtype=float)
        self.status_: str = "init"
        self.error_: Exception | None = None

    def _fit(self, X, y=None):
        return self

    def _predict(self, X, axis=None):
        signal = np.asarray(X, dtype=float)
        if axis is not None and axis != self.axis:
            signal = np.moveaxis(signal, axis, self.axis)
        if signal.ndim == 1:
            signal = signal[:, None]

        signal_len, dim = signal.shape

        error: Exception | None = None
        status = "fail"
        pred: np.ndarray | None = None

        try:
            n_cp = (
                self.n_changepoint_func(signal)
                if self.n_changepoint_func is not None
                else self.n_changepoints
            )

            if dim > 1 and self.multivariate_strategy == "ensembling":
                # Ensembling strategy: fit Prophet on each dimension and aggregate
                ds_index = _safe_date_range(signal_len)
                all_detected_indices = []
                
                for d in range(dim):
                    dim_values = signal[:, d]
                    df = pd.DataFrame({"y": dim_values, "ds": ds_index})
                    
                    model = Prophet(n_changepoints=n_cp, changepoint_range=1.0)
                    model.fit(df)
                    
                    cp_dates = model.changepoints
                    indices = df[df["ds"].isin(cp_dates)].index.to_numpy(dtype=int)
                    all_detected_indices.extend(indices)
                
                pred = aggregate_change_points(all_detected_indices, n_cp, self.tolerance, signal_len=signal_len)
                status = "ok"

            else:
                if dim > 1:
                    # L2 strategy: compute L2 norm across dimensions
                    values = multivariate_l2_norm(signal)
                else:
                    values = signal.flatten()

                df = pd.DataFrame({"y": values})
                df["ds"] = _safe_date_range(len(df))

                model = Prophet(n_changepoints=n_cp, changepoint_range=1.0)
                model.fit(df)

                # Correctly map detected changepoint dates back to indices
                cp_dates = model.changepoints
                # Find indices where ds matches the changepoints
                pred = df[df["ds"].isin(cp_dates)].index.to_numpy(dtype=int)
                status = "ok"
        except Exception as exc:
            error = exc

        self.status_ = status
        self.error_ = error

        if error is not None:
            raise error

        self.changepoints_ = (
            np.asarray(pred, dtype=int) if pred is not None else np.empty(0, dtype=int)
        )
        self.changepoint_strengths_ = np.empty(0, dtype=float)
        return self.changepoints_

