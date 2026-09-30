"""``Pelt._seg`` and ``Dynp.seg`` in numba.

For the kernel costs, a transcription of the C solvers behind ruptures'
``KernelCPD`` (``detection/_detection/ekcpd_pelt_computation.c`` and
``ekcpd_computation.c``): the kernel sums of every candidate segment are
updated one sample at a time, in O(n) memory, where the Python path reads them
off the n x n Gram matrix. For l1 and l2, the costs of the segments that end at
t come from one sweep back from t.

The solvers keep the semantics of the vendored Python ``Pelt`` and ``Dynp``,
not those of the C code, so that both paths return the same segmentation:

* change points on the ``jump`` grid (the C code forces ``jump=1``);
* the kernels of ``CostRbf`` and ``CostCosine``, unclipped (the C code clips
  ``gamma * ||x - y||^2`` to [0.01, 100], diagonal included);
* the pruning of ``Pelt``, delayed by ``min_size`` (upstream PR #383), and the
  admissible change points of ``Dynp``;
* the objective summed segment after segment (the Python solvers sum their
  partition dictionaries, with Python >= 3.12's compensated ``sum``);
* ties broken as the Python solvers break them: the first candidate within
  ``TIE_RTOL`` of the minimum (relative to it, or to the costs of t samples,
  ``t * unit``, when it is close to 0), since costs computed in another order
  than numpy's differ from them by rounding.
"""

from __future__ import annotations

import numpy as np

from ..utils.utils import TIE_RTOL
from ._costs import (
    COSINE,
    RBF,
    acc_new,
    acc_push,
    acc_reset,
    acc_value,
    kernel_extend,
    row_norms,
)
from ._jit import njit

_NEVER = np.iinfo(np.int64).max


@njit(cache=True)
def _sweep_back(x, t, starts, a_lo, a_hi, kind, acc, out):
    """out[a] = cost of [starts[a], t) for a_lo <= a < a_hi, in one sweep of
    the samples from t - 1 down to starts[a_lo] (l1, l2)."""
    acc_reset(acc)
    i = t
    for a in range(a_hi - 1, a_lo - 1, -1):
        while i > starts[a]:
            i -= 1
            acc_push(x, i, t - i, kind, acc)
        out[a] = acc_value(kind, acc)


@njit(cache=True)
def pelt(x, pen, min_size, jump, kind, gamma, unit):
    """Mirror of ``Pelt._seg``; returns the segment ends, or [] if it has none."""
    n, d = x.shape
    kernels = kind >= RBF
    norms = row_norms(x) if kind == COSINE else np.empty(0)
    n_ends = 0
    for k in range(0, n, jump):
        if k >= min_size:
            n_ends += 1
    ends = np.empty(n_ends + 1, dtype=np.int64)
    m = 0
    for k in range(0, n, jump):
        if k >= min_size:
            ends[m] = k
            m += 1
    ends[n_ends] = n
    starts = np.empty(n_ends + 1, dtype=np.int64)
    starts[0] = 0
    starts[1:] = ends[:-1]

    prune_at = np.full(n + 1, _NEVER, dtype=np.int64)
    value = np.full(n + 1, np.inf)
    value[0] = 0.0
    prev = np.full(n + 1, -1, dtype=np.int64)
    diag = np.zeros(n + 1 if kernels else 0)
    S = np.zeros(n + 1 if kernels else 0)
    acc = acc_new(kind, d, 0 if kernels else n)
    cost = np.empty(n_ends + 1)
    cand = np.empty(n_ends + 1)
    n_adm = 0
    lo = 0
    e = 0
    for t in range(1, n + 1):
        if kernels:
            kernel_extend(x, t, starts[lo], kind, gamma, norms, diag, S)
        if t != ends[e]:
            continue
        e += 1
        while n_adm <= n_ends and starts[n_adm] <= t - min_size:
            n_adm += 1
        if not kernels:
            _sweep_back(x, t, starts, lo, n_adm, kind, acc, cost)
        best = np.inf
        for a in range(lo, n_adm):
            s = starts[a]
            if t >= prune_at[s]:
                continue
            if kernels:
                cost[a] = (diag[t] - diag[s]) - S[s] / (t - s)
            cand[a] = value[s] + (cost[a] + pen)
            if cand[a] < best:
                best = cand[a]
        if best == np.inf:
            return np.empty(0, dtype=np.int64)
        limit = best + TIE_RTOL * max(abs(best), t * unit)
        for a in range(lo, n_adm):
            s = starts[a]
            if t < prune_at[s] and cand[a] <= limit:
                best = cand[a]
                prev[t] = s
                break
        value[t] = best
        bound = best + pen
        bound += TIE_RTOL * max(abs(bound), t * unit)
        for a in range(lo, n_adm):
            s = starts[a]
            if prune_at[s] != _NEVER:
                continue
            if cand[a] > bound:
                prune_at[s] = t + min_size
        while lo < n_adm - 1 and prune_at[starts[lo]] <= t:
            lo += 1
    return _backtrack(prev, n)


@njit(cache=True)
def _backtrack(prev, n):
    count = 0
    t = n
    while t > 0:
        count += 1
        t = prev[t]
    out = np.empty(count, dtype=np.int64)
    t = n
    for i in range(count - 1, -1, -1):
        out[i] = t
        t = prev[t]
    return out


@njit(cache=True)
def _sanity(n_samples, n_bkps, jump, min_size):
    """``ruptures.utils.sanity_check``."""
    if n_samples <= 0 or jump <= 0:
        return False
    if n_bkps > n_samples // jump:
        return False
    if n_bkps * ((min_size + jump - 1) // jump) * jump + min_size > n_samples:
        return False
    return True


@njit(cache=True)
def dynp(x, n_bkps, min_size, jump, kind, gamma, unit):
    """Mirror of ``Dynp.seg(0, n, n_bkps)``; returns [] where it would raise."""
    n, d = x.shape
    kernels = kind >= RBF
    norms = row_norms(x) if kind == COSINE else np.empty(0)
    K = n_bkps
    value = np.full((K + 1, n + 1), np.inf)
    fails = np.zeros((K + 1, n + 1), dtype=np.bool_)
    arg = np.full((K + 1, n + 1), -1, dtype=np.int64)
    diag = np.zeros(n + 1 if kernels else 0)
    S = np.zeros(n + 1 if kernels else 0)
    # l1, l2: grid[a] = a * jump, the starts whose cost the sweep records
    grid = np.arange(0, n, jump)
    grid_cost = np.empty(grid.shape[0])
    acc = acc_new(kind, d, 0 if kernels else n)
    cand = np.empty(n + 1)
    first = ((min_size + jump - 1) // jump) * jump  # smallest change point
    for t in range(1, n + 1):
        if kernels:
            kernel_extend(x, t, 0, kind, gamma, norms, diag, S)
        if t % jump != 0 and t != n:
            continue
        if kernels:
            value[0, t] = (diag[t] - diag[0]) - S[0] / t
        else:
            _sweep_back(x, t, grid, 0, (t - 1) // jump + 1, kind, acc, grid_cost)
            value[0, t] = grid_cost[0]
        for k in range(1, K + 1):
            best = np.inf
            found = False
            for b in range(first, t - min_size + 1, jump):
                if not _sanity(b, k - 1, jump, min_size):
                    continue
                found = True
                if fails[k - 1, b]:
                    fails[k, t] = True
                if kernels:
                    cost = (diag[t] - diag[b]) - S[b] / (t - b)
                else:
                    cost = grid_cost[b // jump]
                cand[b] = value[k - 1, b] + cost
                if cand[b] < best:
                    best = cand[b]
            if not found:
                fails[k, t] = True
                continue
            limit = best + TIE_RTOL * max(abs(best), t * unit)
            for b in range(first, t - min_size + 1, jump):
                if _sanity(b, k - 1, jump, min_size) and cand[b] <= limit:
                    value[k, t] = cand[b]
                    arg[k, t] = b
                    break
    if fails[K, n]:
        return np.empty(0, dtype=np.int64)
    out = np.empty(K + 1, dtype=np.int64)
    out[K] = n
    t = n
    for k in range(K, 0, -1):
        t = arg[k, t]
        out[k - 1] = t
    return out
