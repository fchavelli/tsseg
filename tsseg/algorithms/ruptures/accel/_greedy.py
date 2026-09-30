"""The inner loops of ``Binseg`` and ``Window`` in numba (``BottomUp`` only
needs ``segment_cost``)."""

from __future__ import annotations

import numpy as np

from ..utils.utils import TIE_RTOL
from ._costs import RBF, acc_new, acc_push, acc_reset, acc_value, kernel, segment_cost
from ._jit import njit


@njit(cache=True)
def _kernel_sweeps(x, start, end, kind, gamma, norms, left, right):
    """Both sweeps of ``_sweeps`` for a kernel cost, in one pass over the pairs:
    ``col[b]`` sums k(x_a, x_b) over a < b, ``row[a]`` over b > a, each in the
    order of the sweep that needs it, so half the kernel evaluations give the
    same sums."""
    n = end - start
    col = np.zeros(n)
    row = np.zeros(n)
    diag = np.empty(n)
    for a in range(n):
        i = start + a
        diag[a] = kernel(x, i, i, kind, gamma, norms)
        acc = 0.0
        for b in range(a + 1, n):
            k = kernel(x, i, start + b, kind, gamma, norms)
            acc += k
            col[b] += k
        row[a] = acc
    within = 0.0
    trace = 0.0
    for b in range(1, n + 1):
        within += 2.0 * col[b - 1] + diag[b - 1]
        trace += diag[b - 1]
        left[b] = trace - within / b
    within = 0.0
    trace = 0.0
    for a in range(n - 1, -1, -1):
        within += 2.0 * row[a] + diag[a]
        trace += diag[a]
        right[a] = trace - within / (n - a)


@njit(cache=True)
def _sweeps(x, start, end, kind, gamma, norms, left, right):
    """left[i] = cost of [start, start + i) for 0 < i <= n, right[i] = cost of
    [start + i, end) for 0 <= i < n, n = end - start: one sweep from each end."""
    if kind >= RBF:
        _kernel_sweeps(x, start, end, kind, gamma, norms, left, right)
        return
    n = end - start
    acc = acc_new(kind, x.shape[1], n)
    for b in range(start + 1, end + 1):
        acc_push(x, b - 1, b - start, kind, acc)
        left[b - start] = acc_value(kind, acc)
    acc_reset(acc)
    for s in range(end - 1, start - 1, -1):
        acc_push(x, s, end - s, kind, acc)
        right[s - start] = acc_value(kind, acc)


@njit(cache=True)
def binseg_split(x, start, end, min_size, jump, kind, gamma, norms, unit):
    """Mirror of ``Binseg.single_bkp``: the best change point of [start, end)
    (-1 if none is admissible), its gain and the cost of [start, end).

    A sweep from each end gives the costs of all the [start, b) and [b, end)
    in O(n) (l2), O(n log n) (l1) or O(n^2) (kernels) instead of O(n) costs of
    O(n), O(n log n) or O(n^2) each. Gains within ``TIE_RTOL`` of the segment's
    cost (or of ``n * unit``, the typical cost of its n samples) are tied, and
    the last change point wins, as ``max`` over (gain, change point) pairs
    picks among exact ties.
    """
    n = end - start
    left = np.zeros(n + 1)
    right = np.zeros(n + 1)
    _sweeps(x, start, end, kind, gamma, norms, left, right)
    whole = left[n]
    best = -np.inf
    for b in range(start, end, jump):
        if b - start >= min_size and end - b >= min_size:
            gain = whole - left[b - start] - right[b - start]
            if gain > best:
                best = gain
    if best == -np.inf:
        return -1, 0.0, whole
    limit = best - TIE_RTOL * max(abs(whole), n * unit)
    bkp = -1
    bkp_gain = 0.0
    for b in range(start, end, jump):
        if b - start >= min_size and end - b >= min_size:
            gain = whole - left[b - start] - right[b - start]
            if gain >= limit:
                bkp = b
                bkp_gain = gain
    return bkp, bkp_gain, whole


@njit(cache=True)
def window_scores(x, inds, width, kind, gamma, norms):
    """``Window.fit``'s scores, cost(window) - (cost(left half) + cost(right
    half)) around each index, and cost(window), the scale of their rounding."""
    half = width // 2
    buf = np.empty(width * x.shape[1])
    score = np.empty(inds.shape[0])
    scale = np.empty(inds.shape[0])
    for k in range(inds.shape[0]):
        idx = inds[k]
        start = idx - half
        end = idx + half
        whole = segment_cost(x, start, end, kind, gamma, norms, buf)
        parts = segment_cost(x, start, idx, kind, gamma, norms, buf)
        parts += segment_cost(x, idx, end, kind, gamma, norms, buf)
        score[k] = whole - parts
        scale[k] = whole
    return score, scale
