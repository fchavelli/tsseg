"""The median heuristic of ``CostRbf`` without the n x n distance matrix."""

from __future__ import annotations

import numpy as np

from ._costs import _sq_dist
from ._jit import njit, prange


@njit(cache=True, parallel=True)
def _radix_pass(x, shift, prefix, n_chunks):
    """Histogram, over the pairs i < j with a positive squared distance d
    whose bits above ``shift + 16`` equal ``prefix``, of bits
    ``shift .. shift + 15`` of d. Positive doubles sort as their bit patterns.
    """
    n = x.shape[0]
    hist = np.zeros((n_chunks, 1 << 16), dtype=np.int64)
    for c in prange(n_chunks):
        buf = np.empty(1)
        bits = buf.view(np.int64)
        for i in range(c, n, n_chunks):
            for j in range(i + 1, n):
                d = _sq_dist(x, i, j)
                if d > 0.0:
                    buf[0] = d
                    b = bits[0]
                    if shift < 48 and (b >> (shift + 16)) != prefix:
                        continue
                    hist[c, (b >> shift) & 0xFFFF] += 1
    return hist.sum(axis=0)


@njit(cache=True, parallel=True)
def _around(x, v, n_chunks):
    """Count of positive squared distances <= v and smallest one above v."""
    n = x.shape[0]
    count = np.zeros(n_chunks, dtype=np.int64)
    above = np.full(n_chunks, np.inf)
    for c in prange(n_chunks):
        for i in range(c, n, n_chunks):
            for j in range(i + 1, n):
                d = _sq_dist(x, i, j)
                if d > 0.0:
                    if d <= v:
                        count[c] += 1
                    elif d < above[c]:
                        above[c] = d
    return count.sum(), above.min()


def _select(x, rank, n_chunks, top):
    """Exact ``rank``-th smallest positive squared distance: a radix select
    over the bit pattern, 16 bits per pass. ``top`` is the first pass."""
    prefix = 0
    for shift in (48, 32, 16, 0):
        hist = top if shift == 48 else _radix_pass(x, shift, prefix, n_chunks)
        below = np.cumsum(hist)
        digit = int(np.searchsorted(below, rank, side="right"))
        if digit > 0:
            rank -= int(below[digit - 1])
        prefix = (prefix << 16) | digit
    return float(np.array([prefix], dtype=np.int64).view(np.float64)[0])


def median_sq_dist(x: np.ndarray) -> float:
    """Median of the positive pairwise squared distances between the rows of
    ``x``, as ``np.median(d[d > 0])`` with ``d = pdist(x, "sqeuclidean")``,
    in O(n) memory. 0.0 when every row is identical.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    n_chunks = 64
    top = _radix_pass(x, 48, 0, n_chunks)
    total = int(top.sum())
    if total == 0:
        return 0.0
    low = _select(x, (total - 1) // 2, n_chunks, top)
    if total % 2:
        return low
    n_le, above = _around(x, low, n_chunks)
    high = low if n_le > total // 2 else float(above)
    return (low + high) / 2.0
