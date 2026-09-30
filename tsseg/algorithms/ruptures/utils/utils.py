"""Miscellaneous helpers used by the vendored ruptures estimators."""

from __future__ import annotations

from itertools import tee
from math import ceil, isfinite, log

import numpy as np

from ..exceptions import NotEnoughPoints

#: Values closer than this, relative to the magnitude of the quantities they are
#: computed from, are tied: the solvers then decide by position, as they decide
#: between exactly equal values. The same cost summed in two orders differs by
#: rounding, orders of magnitude below this gap, so the Python and numba paths
#: take the same decisions.
TIE_RTOL = 1e-9


def tie_tol(*scales: float) -> float:
    """Gap below which two values computed from quantities of magnitude
    ``scales`` are tied; non-finite scales are ignored."""

    return TIE_RTOL * max([0.0, *(abs(s) for s in scales if isfinite(s))])


def tie_limit(best: float, floor: float = 0.0) -> float:
    """Largest value still tied with the minimum ``best``. ``floor``: the
    magnitude of the costs summed into these values, which sets the gap when
    ``best`` is close to 0."""

    return best + tie_tol(best, floor)


def tie_unit(cost) -> float:
    """Typical cost of one sample of the fitted ``cost``'s signal: values
    computed over n samples have a tie gap of at least ``tie_tol(n * unit)``.

    The kernel costs are bounded by the kernel's diagonal, k(x, x) <= 1. For
    the others, the cost of the whole signal, and for l2 at least 1e-16 of the
    sum of x^2: numpy's l2 cost of a constant segment at x is not 0 but up to
    (c * eps * x)^2 per sample, its mean being off by c * eps * |x| (c < 1400
    below 10^7 samples), and must tie with the 0 of the numba sweeps.
    """

    if cost.model in ("rbf", "cosine"):
        return 1.0
    signal = cost.signal
    n = signal.shape[0]
    try:
        whole = abs(float(cost.error(0, n)))
    except NotEnoughPoints:
        return 0.0
    if not isfinite(whole):
        return 0.0
    if cost.model == "l2":
        whole = max(whole, 1e-16 * float(np.square(signal).sum()))
    return whole / n


def bic_penalty(coefficient: float, cost) -> float:
    """``coefficient * log(n) * tie_unit(cost)``: a penalty in units of the
    typical cost of one sample of the fitted ``cost``'s signal.

    For l2 on unit-variance channels, the unit is d and ``log(n) * d`` is the
    BIC penalty of one more segment (d means); for l1, the absolute deviations
    from the median per sample, summed over the channels; for the kernel costs,
    bounded by 1 per sample, 1. One coefficient then applies across lengths,
    dimensions, scales and costs. A cost of 0 (a constant signal) gives the
    unit 1: any positive penalty then returns the signal whole.
    """

    unit = tie_unit(cost)
    n = cost.signal.shape[0]
    return float(coefficient) * log(max(n, 2)) * (unit if unit > 0 else 1.0)


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
