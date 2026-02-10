"""ESPRESSO change-point detector implemented purely in Python."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base import BaseSegmenter
from .python.ESPRESSO_Script import espresso as _run_espresso

__all__ = ["EspressoDetector"]


def _ensure_time_major(X: Any, axis: int) -> np.ndarray:
    """Return a 2D time-major view of ``X`` while preserving numeric type."""

    data = np.asarray(X, dtype=float)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    if axis != 0:
        data = np.moveaxis(data, axis, 0)
    if data.ndim != 2:
        raise ValueError("EspressoDetector expects 1D or 2D inputs")
    return data


class EspressoDetector(BaseSegmenter):
    """Segment time series using the ESPRESSO change-point detection algorithm.

    Parameters
    ----------
    subsequence_length : int, default=64
        Length of the subsequences used to compute the matrix profile; must be
        at least 4.
    chain_len : int, default=3
        Number of iterations used when expanding arc sets to build the
        semantic density matrix.
    n_segments : int, optional
        Target number of segments to produce during prediction. Must be
        supplied and be greater than or equal to 2 when calling ``predict``.
    peak_distance_fraction : float, default=0.01
        Fraction of the input length that enforces a minimum spacing between
        detected peaks (matches the MATLAB reference implementation).
    axis : int, default=0
        Axis corresponding to the time dimension in the input array.
    random_state : int, optional
        Seed for the internal random number generator used when sampling
        subsequences for the matrix profile.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "semi_supervised": True,
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        window_size: int = 64,
        chain_len: int = 3,
        *,
        n_segments: int | None = None,
        peak_distance_fraction: float = 0.01,
        axis: int = 0,
        random_state: int | None = None,
    ) -> None:
        if window_size < 4:
            raise ValueError("window_size must be >= 4")
        if peak_distance_fraction <= 0:
            raise ValueError("peak_distance_fraction must be strictly positive")

        self.window_size = int(window_size)
        self.chain_len = int(chain_len)
        self.n_segments = n_segments
        self.peak_distance_fraction = float(peak_distance_fraction)
        self.random_state = random_state
        self._rng: np.random.Generator | None = None
        self._fitted_change_points: np.ndarray | None = None

        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # BaseSegmenter overrides
    # ------------------------------------------------------------------

    def _fit(self, X, y=None):
        self._fitted_change_points = None
        if self.n_segments is None:
            raise ValueError(
                "For semi-supervised fitting, `n_segments` must be provided "
                "at initialization."
            )
        return self

    def _predict(self, X, axis=None):
        axis = self.axis if axis is None else axis
        signal = _ensure_time_major(X, axis=axis)

        if self.n_segments is None or self.n_segments < 2:
            raise ValueError("n_segments must be provided and >= 2 for prediction")


        # ``espresso`` expects channels x time ordering.
        data = signal.T
        change_points = _run_espresso(
            self.n_segments,
            data,
            self.window_size,
            self.chain_len,
            pdist_fraction=self.peak_distance_fraction,
            rng=self._rng,
        )

        change_points = np.asarray(change_points, dtype=int)
        if change_points.size:
            n_timepoints = signal.shape[0]
            change_points = change_points[(0 < change_points) & (change_points < n_timepoints)]
            change_points = np.unique(change_points)

        self._fitted_change_points = change_points.copy()
        return change_points

    # ------------------------------------------------------------------
    # Public API extensions
    # ------------------------------------------------------------------

    def get_fitted_params(self):
        change_points = None
        if self._fitted_change_points is not None:
            change_points = self._fitted_change_points.copy()
        return {
            "change_points": change_points,
            "n_segments": self.n_segments,
            "window_size": self.window_size,
            "chain_len": self.chain_len,
            "peak_distance_fraction": self.peak_distance_fraction,
        }