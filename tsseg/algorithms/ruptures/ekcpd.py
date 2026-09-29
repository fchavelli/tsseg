"""Kernel change point solvers that never build the Gram matrix.

Numba transcription of the C solvers behind ruptures' ``KernelCPD``
(``detection/_detection/ekcpd_pelt_computation.c`` and
``ekcpd_computation.c``): the kernel sums of every candidate segment are
updated one sample at a time, in O(n) memory, where ``CostRbf`` and
``CostCosine`` read them off an n x n Gram matrix.

The solvers keep the semantics of the vendored Python ``Pelt`` and ``Dynp``,
not those of the C code, so that both paths return the same segmentation:

* change points on the ``jump`` grid (the C code forces ``jump=1``);
* the kernels of ``CostRbf`` and ``CostCosine``, unclipped (the C code clips
  ``gamma * ||x - y||^2`` to [0.01, 100], diagonal included);
* the pruning of ``Pelt``, delayed by ``min_size`` (upstream PR #383), and the
  admissible change points of ``Dynp``;
* ties broken as the Python solvers break them: the first candidate within
  ``TIE_RTOL`` of the minimum, since sums accumulated in another order than
  numpy's differ from them by rounding.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    AVAILABLE = True
except ImportError:  # numba is optional (tsseg[accelerators])
    AVAILABLE = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda func: func


from .utils.utils import TIE_RTOL  # noqa: E402

RBF = 0
COSINE = 1
#: Cost models these solvers handle, by ``BaseCost.model``.
KERNELS = {"rbf": RBF, "cosine": COSINE}

_NEVER = np.iinfo(np.int64).max


@njit(cache=True)
def _sq_dist(x, i, j):
    s = 0.0
    for c in range(x.shape[1]):
        diff = x[i, c] - x[j, c]
        s += diff * diff
    return s


@njit(cache=True)
def _row_norms(x):
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        s = 0.0
        for c in range(x.shape[1]):
            s += x[i, c] * x[i, c]
        out[i] = np.sqrt(s)
    return out


@njit(cache=True)
def _kernel(x, i, j, kind, gamma, norms):
    if kind == RBF:
        if i == j:
            return 1.0
        return np.exp(-gamma * _sq_dist(x, i, j))
    den = norms[i] * norms[j]
    if den > 0.0:
        dot = 0.0
        for c in range(x.shape[1]):
            dot += x[i, c] * x[j, c]
        return dot / den
    return 0.0


@njit(cache=True)
def _add_sample(x, t, lo, kind, gamma, norms, diag, S):
    """Extend every segment [s, t - 1) with s >= lo by the sample t - 1.

    ``diag[t]`` is the prefix sum of k(x_i, x_i) and ``S[s]`` the sum of
    k(x_i, x_j) over [s, t)^2 once the call returns.
    """
    k_tt = _kernel(x, t - 1, t - 1, kind, gamma, norms)
    diag[t] = diag[t - 1] + k_tt
    acc = 0.0
    for s in range(t - 1, lo - 1, -1):
        acc += _kernel(x, s, t - 1, kind, gamma, norms)
        S[s] += 2.0 * acc - k_tt


@njit(cache=True)
def _pelt(x, pen, min_size, jump, kind, gamma):
    """Mirror of ``Pelt._seg``; returns the segment ends, or [] if it has none."""
    n = x.shape[0]
    norms = _row_norms(x) if kind == COSINE else np.empty(0)
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
    diag = np.zeros(n + 1)
    S = np.zeros(n + 1)
    cost = np.empty(n_ends + 1)
    cand = np.empty(n_ends + 1)
    n_adm = 0
    lo = 0
    e = 0
    for t in range(1, n + 1):
        _add_sample(x, t, starts[lo], kind, gamma, norms, diag, S)
        if t != ends[e]:
            continue
        e += 1
        while n_adm <= n_ends and starts[n_adm] <= t - min_size:
            n_adm += 1
        best = np.inf
        for a in range(lo, n_adm):
            s = starts[a]
            if t >= prune_at[s]:
                continue
            cost[a] = (diag[t] - diag[s]) - S[s] / (t - s)
            cand[a] = value[s] + (cost[a] + pen)
            if cand[a] < best:
                best = cand[a]
        if best == np.inf:
            return np.empty(0, dtype=np.int64)
        limit = best + TIE_RTOL * max(1.0, abs(best))
        for a in range(lo, n_adm):
            s = starts[a]
            if t < prune_at[s] and cand[a] <= limit:
                best = cand[a]
                prev[t] = s
                break
        value[t] = best
        for a in range(lo, n_adm):
            s = starts[a]
            if prune_at[s] != _NEVER:
                continue
            if cand[a] > best + pen:
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
def _dynp(x, n_bkps, min_size, jump, kind, gamma):
    """Mirror of ``Dynp.seg(0, n, n_bkps)``; returns [] where it would raise."""
    n = x.shape[0]
    norms = _row_norms(x) if kind == COSINE else np.empty(0)
    K = n_bkps
    value = np.full((K + 1, n + 1), np.inf)
    fails = np.zeros((K + 1, n + 1), dtype=np.bool_)
    arg = np.full((K + 1, n + 1), -1, dtype=np.int64)
    diag = np.zeros(n + 1)
    S = np.zeros(n + 1)
    cand = np.empty(n + 1)
    first = ((min_size + jump - 1) // jump) * jump  # smallest change point
    for t in range(1, n + 1):
        _add_sample(x, t, 0, kind, gamma, norms, diag, S)
        if t % jump != 0 and t != n:
            continue
        value[0, t] = (diag[t] - diag[0]) - S[0] / t
        for k in range(1, K + 1):
            best = np.inf
            found = False
            for b in range(first, t - min_size + 1, jump):
                if not _sanity(b, k - 1, jump, min_size):
                    continue
                found = True
                if fails[k - 1, b]:
                    fails[k, t] = True
                cand[b] = value[k - 1, b] + ((diag[t] - diag[b]) - S[b] / (t - b))
                if cand[b] < best:
                    best = cand[b]
            if not found:
                fails[k, t] = True
                continue
            limit = best + TIE_RTOL * max(1.0, abs(best))
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


BACKENDS = ("auto", "numba", "python")


def use_numba(cost, backend: str) -> bool:
    """Whether a solver runs ``cost`` here rather than in Python.

    ``"auto"`` does for ``CostRbf`` and ``CostCosine`` when numba is installed,
    ``"numba"`` insists (and raises otherwise), ``"python"`` never does.
    """
    from .costs import CostCosine, CostRbf

    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "python":
        return False
    handled = type(cost) in (CostRbf, CostCosine)
    if backend == "numba":
        if not AVAILABLE:
            raise ImportError(
                "backend='numba' needs numba (pip install tsseg[accelerators])"
            )
        if not handled:
            raise ValueError(
                f"backend='numba' handles the costs {sorted(KERNELS)}, "
                f"not {cost.model!r}"
            )
    return handled and AVAILABLE


def _kernel_of(cost) -> tuple[int, float]:
    from .costs import CostRbf

    if isinstance(cost, CostRbf):
        return RBF, float(cost.gamma)
    return COSINE, 1.0


def pelt(cost, pen: float, min_size: int, jump: int) -> list[int]:
    """``Pelt._seg`` on a fitted ``CostRbf`` / ``CostCosine``: the segment ends."""
    kind, gamma = _kernel_of(cost)
    x = np.ascontiguousarray(cost.signal, dtype=np.float64)
    ends = _pelt(x, float(pen), int(min_size), int(jump), kind, gamma)
    if ends.size == 0:
        raise ValueError("PELT found no admissible segmentation")
    return [int(e) for e in ends]


def dynp(cost, n_bkps: int, min_size: int, jump: int) -> list[int]:
    """``Dynp.seg(0, n, n_bkps)`` on a fitted kernel cost: the segment ends."""
    from .exceptions import BadSegmentationParameters

    kind, gamma = _kernel_of(cost)
    x = np.ascontiguousarray(cost.signal, dtype=np.float64)
    ends = _dynp(x, int(n_bkps), int(min_size), int(jump), kind, gamma)
    if ends.size == 0:
        raise BadSegmentationParameters
    return [int(e) for e in ends]
