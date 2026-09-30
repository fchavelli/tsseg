"""Lightweight alternatives to SciPy peak utilities."""

from __future__ import annotations

import numpy as np


def argrelmax_1d(
    values: np.ndarray, order: int, tol: np.ndarray | None = None
) -> np.ndarray:
    """Return indices of relative maxima within ``values``.

    Mirrors ``scipy.signal.argrelmax(values, order=order, mode="wrap")``, the
    call of upstream ruptures' ``Window``: a maximum exceeds the ``order``
    values on each side of it, the array wrapping around at its ends (an
    ``order`` of at least ``values.size`` compares a value with itself, so that
    there is no maximum).

    ``tol``: per-value tie tolerances. A maximum must then exceed each of its
    neighbours by more than the larger of their two tolerances, so that values
    equal up to rounding form a plateau, which has no maximum.
    """

    if order < 1:
        raise ValueError("order must be a positive integer")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    n = values.size
    if n == 0:
        return np.array([], dtype=int)

    if tol is None:
        tol = np.zeros(n)
    locs = np.arange(n)
    maxima = np.ones(n, dtype=bool)
    for shift in range(1, order + 1):
        for other in (locs + shift, locs - shift):
            gap = np.maximum(tol, np.take(tol, other, mode="wrap"))
            maxima &= values > np.take(values, other, mode="wrap") + gap
    return np.flatnonzero(maxima)
