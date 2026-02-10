"""Lightweight alternatives to SciPy peak utilities."""

from __future__ import annotations

import numpy as np


def argrelmax_1d(values: np.ndarray, order: int) -> np.ndarray:
    """Return indices of relative maxima within ``values``.

    This minimal implementation mirrors :func:`scipy.signal.argrelmax` for the
    one-dimensional case which is sufficient for the window detector.
    """

    if order <= 0:
        raise ValueError("order must be a positive integer")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    n = values.size
    if n == 0:
        return np.array([], dtype=int)

    order = min(order, max(1, n - 1))
    maxima = []
    for idx in range(n):
        left = max(0, idx - order)
        right = min(n, idx + order + 1)
        current = values[idx]
        if np.all(current > values[left:idx]) and np.all(current > values[idx + 1 : right]):
            maxima.append(idx)
    return np.asarray(maxima, dtype=int)
