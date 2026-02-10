"""Miscellaneous helpers used by the vendored ruptures estimators."""

from __future__ import annotations

from itertools import tee
from math import ceil


def pairwise(iterable):
    """Yield consecutive pairs from ``iterable``."""

    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unzip(sequence):
    """Inverse of :func:`zip` returning a tuple of iterables."""

    return zip(*sequence)


def sanity_check(n_samples: int, n_bkps: int, jump: int, min_size: int) -> bool:
    """Return ``True`` when segmentation parameters admit a solution."""

    if n_samples <= 0:
        return False
    if jump <= 0:
        return False
    n_admissible = n_samples // jump
    if n_bkps > n_admissible:
        return False
    if n_bkps * ceil(min_size / jump) * jump + min_size > n_samples:
        return False
    return True
