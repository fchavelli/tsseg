"""Miscellaneous helpers used by the vendored ruptures estimators."""

from __future__ import annotations

from itertools import tee
from math import ceil


#: Candidates within this relative gap of the minimum are tied, and the first of
#: them (the earliest change point) wins. The same segmentation computed in two
#: summation orders differs by rounding, orders of magnitude below this gap, so
#: the Python and numba solvers break exact ties alike.
TIE_RTOL = 1e-9


def tie_limit(best: float) -> float:
    """Largest value still tied with the minimum ``best``."""

    return best + TIE_RTOL * max(1.0, abs(best))


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
