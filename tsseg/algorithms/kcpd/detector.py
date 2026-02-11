"""Kernel-based change point detector built on the vendored ruptures implementation."""

from __future__ import annotations

import warnings

import numpy as np

from ..base import BaseSegmenter
from ..ruptures.detection import KernelCPD

__all__ = ["KCPDDetector"]


class KCPDDetector(BaseSegmenter):
    """Kernel change point detector using dynamic programming or PELT."""

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
        pen: float | None = 10,
        kernel: str = "rbf",
        min_size: int = 2,
        jump: int = 1,
        cost_params: dict | None = None,
        axis: int = 0,
    ) -> None:
        self.n_cps = None if n_cps is None else int(n_cps)
        self.pen = None if pen is None else float(pen)
        has_n_cps = n_cps is not None
        has_pen = pen is not None
        if not (has_n_cps or has_pen):
            raise ValueError("Configure either n_cps or pen")

        if has_n_cps and has_pen:
            warnings.warn(
                "pen is ignored when n_cps is provided; proceeding with n_cps only",
                UserWarning,
            )
            self.pen = None
        self.kernel = kernel
        self.min_size = int(min_size)
        self.jump = int(jump)
        self.cost_params = cost_params or {}
        self._estimator: KernelCPD | None = None
        self._train_signal: np.ndarray | None = None
        super().__init__(axis=axis)

    def _ensure_2d(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            return array[:, np.newaxis]
        if array.ndim != 2:
            raise ValueError("KernelCPDDetector expects 1D or 2D arrays")
        return array

    def _fit(self, X, y=None):
        signal = self._ensure_2d(X)
        estimator = KernelCPD(
            kernel=self.kernel,
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
            raise RuntimeError("KernelCPDDetector must be fitted before predict")
        signal = self._ensure_2d(X)
        if self._train_signal is None or not np.array_equal(signal, self._train_signal):
            self._estimator.fit(signal)
            self._train_signal = signal
        bkps = self._estimator.predict(n_bkps=self.n_cps, pen=self.pen)
        bkps = np.asarray(bkps, dtype=int)
        bkps = bkps[(bkps > 0) & (bkps < signal.shape[0])]
        return np.unique(bkps)
