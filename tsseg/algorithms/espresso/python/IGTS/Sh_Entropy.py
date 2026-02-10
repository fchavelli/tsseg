"""Python port of ``Sh_Entropy.m``."""

from __future__ import annotations

import numpy as np

__all__ = ["sh_entropy"]


def sh_entropy(x: np.ndarray) -> float:
    """Compute Shannon entropy of a numeric vector.

    Zero entries are ignored to mirror the MATLAB implementation.
    """

    values = np.asarray(x, dtype=float).reshape(-1)
    values = values[values != 0]
    if values.size == 0:
        return 0.0

    total = float(values.sum())
    if total == 0.0:
        return 0.0

    probs = values / total
    return float(-np.sum(probs * np.log(probs)))
