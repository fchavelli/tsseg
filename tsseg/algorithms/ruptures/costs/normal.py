"""Gaussian likelihood cost."""

from __future__ import annotations

import numpy as np
from numpy.linalg import slogdet

from ..base import BaseCost
from ..exceptions import NotEnoughPoints

RIDGE = 1e-6


class CostNormal(BaseCost):
    """Cost based on the Gaussian log-likelihood.

    The cost of a segment of m samples is ``m * log det(S + r * I)``, with S the
    maximum likelihood (``ddof=0``) covariance of the segment: -2 times its
    Gaussian log-likelihood, up to a constant in m. The ridge r is ``1e-6``
    times the mean variance of the channels of the whole signal (``1e-6`` for a
    constant signal), so rescaling the signal by s moves every segment cost by
    ``2 * d * m * log(s)``, a total that does not depend on the partition: the
    change points do not depend on the scale.

    A segment of at most d samples has a singular covariance, whose cost the
    ridge alone would set: once fitted on d channels, ``min_size`` is
    ``max(2, d + 1)``.
    """

    model = "normal"

    def __init__(self, add_small_diag: bool = True) -> None:
        self.signal: np.ndarray | None = None
        self.add_small_diag = add_small_diag
        self.min_size = 2
        self._ridge = 0.0

    def fit(self, signal: np.ndarray) -> CostNormal:
        signal = np.asarray(signal, dtype=np.float64)  # whatever the input dtype
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        self.signal = signal
        n_channels = signal.shape[1]
        self.min_size = max(2, n_channels + 1)
        self._ridge = 0.0
        if self.add_small_diag:
            scale = float(np.var(signal, axis=0).mean()) if signal.shape[0] else 0.0
            self._ridge = RIDGE * (scale if scale > 0 and np.isfinite(scale) else 1.0)
        return self

    def error(self, start: int, end: int) -> float:
        if self.signal is None:
            raise RuntimeError("Cost not fitted")
        if end - start < self.min_size:
            raise NotEnoughPoints
        segment = self.signal[start:end]
        cov = np.atleast_2d(np.cov(segment.T, bias=True))
        if self._ridge:
            cov = cov + self._ridge * np.eye(cov.shape[0])
        sign, val = slogdet(cov)
        if sign <= 0:
            return float("inf")
        return float(val * (end - start))
