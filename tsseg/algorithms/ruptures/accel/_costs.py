"""Segment costs of ``CostL1``, ``CostL2``, ``CostRbf`` and ``CostCosine``.

Two ways to evaluate them:

* ``segment_cost``: one segment, from scratch. For l1 and l2 it repeats numpy's
  arithmetic, summation order included (``np_sum``), so that on a C-contiguous
  float64 signal it returns the costs of the Python path bit for bit;
* sweeps (``acc_*``, ``kernel_extend``): one sample at a time, for all the
  segments that share a start or an end, at O(1) per segment for l2 (Welford),
  O(log n) for l1 (two heaps) and O(n) for the kernels, where ``segment_cost``
  takes O(n), O(n log n) and O(n^2). They differ from the Python costs by
  rounding, which the tie rule of the solvers absorbs.
"""

from __future__ import annotations

import numpy as np

from ._jit import njit

L1 = 0
L2 = 1
RBF = 2
COSINE = 3

# numpy sums float64 arrays pairwise within buffers of this many elements
# (NPY_BUFSIZE) and adds up the buffers in sequence.
_NPY_BUFSIZE = 8192


@njit(cache=True)
def _sq_dist(x, i, j):
    s = 0.0
    for c in range(x.shape[1]):
        diff = x[i, c] - x[j, c]
        s += diff * diff
    return s


@njit(cache=True)
def row_norms(x):
    out = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        s = 0.0
        for c in range(x.shape[1]):
            s += x[i, c] * x[i, c]
        out[i] = np.sqrt(s)
    return out


@njit(cache=True)
def kernel(x, i, j, kind, gamma, norms):
    """Entry (i, j) of the Gram matrix of ``CostRbf`` or ``CostCosine``."""
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
def _pairwise_leaf(a, lo, n):
    """numpy's ``pairwise_sum`` of at most 128 values: eight accumulators."""
    if n < 8:
        res = 0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    r0 = a[lo]
    r1 = a[lo + 1]
    r2 = a[lo + 2]
    r3 = a[lo + 3]
    r4 = a[lo + 4]
    r5 = a[lo + 5]
    r6 = a[lo + 6]
    r7 = a[lo + 7]
    i = lo + 8
    stop = lo + n - n % 8
    while i < stop:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < lo + n:
        res += a[i]
        i += 1
    return res


@njit(cache=True)
def _pairwise(a, lo, n):
    """numpy's ``pairwise_sum``: split in halves (rounded down to a multiple
    of 8) down to 128 values. Depth first with an explicit stack, since numba
    does not cache recursive functions."""
    if n <= 128:
        return _pairwise_leaf(a, lo, n)
    first = np.empty(64, dtype=np.int64)
    size = np.empty(64, dtype=np.int64)
    left = np.empty(64)
    right_next = np.zeros(64, dtype=np.bool_)
    top = 0
    first[0] = lo
    size[0] = n
    while True:
        if size[top] > 128:  # descend into the left half
            half = size[top] // 2
            half -= half % 8
            first[top + 1] = first[top]
            size[top + 1] = half
            right_next[top + 1] = False
            top += 1
            continue
        value = _pairwise_leaf(a, first[top], size[top])
        top -= 1
        while top >= 0 and right_next[top]:  # both halves done
            value = left[top] + value
            top -= 1
        if top < 0:
            return value
        left[top] = value  # left half done: the right one
        right_next[top] = True
        half = size[top] // 2
        half -= half % 8
        first[top + 1] = first[top] + half
        size[top + 1] = size[top] - half
        right_next[top + 1] = False
        top += 1


@njit(cache=True)
def np_sum(a, lo, n):
    """``np.add.reduce(a[lo:lo + n])``, bit for bit."""
    total = 0.0
    for start in range(lo, lo + n, _NPY_BUFSIZE):
        total += _pairwise(a, start, min(_NPY_BUFSIZE, lo + n - start))
    return total


@njit(cache=True)
def _l2_segment(x, s, e, buf):
    """``x[s:e].var(axis=0).sum() * (e - s)`` in numpy's order: over axis 0, a
    C-contiguous array is summed row after row, a single column pairwise."""
    n = e - s
    d = x.shape[1]
    if d == 1:
        for i in range(n):
            buf[i] = x[s + i, 0]
        mean = np_sum(buf, 0, n) / n
        for i in range(n):
            dev = buf[i] - mean
            buf[i] = dev * dev
        return (np_sum(buf, 0, n) / n) * n
    for c in range(d):
        acc = 0.0
        for i in range(s, e):
            acc += x[i, c]
        mean = acc / n
        acc = 0.0
        for i in range(s, e):
            dev = x[i, c] - mean
            acc += dev * dev
        buf[c] = acc / n
    return np_sum(buf, 0, d) * n


@njit(cache=True)
def _l1_segment(x, s, e, buf):
    """``abs(x[s:e] - median(x[s:e], axis=0)).sum()`` in numpy's order: the
    median is exact, the sum runs over the deviations in row-major order."""
    n = e - s
    d = x.shape[1]
    med = np.empty(d)
    for c in range(d):
        col = buf[:n]
        for i in range(n):
            col[i] = x[s + i, c]
        col.sort()
        half = n // 2
        if n % 2:
            med[c] = col[half]
        else:
            med[c] = (col[half - 1] + col[half]) / 2.0
    for i in range(n):
        for c in range(d):
            buf[i * d + c] = abs(x[s + i, c] - med[c])
    return np_sum(buf, 0, n * d)


@njit(cache=True)
def _kernel_segment(x, s, e, kind, gamma, norms):
    """Trace minus sum / (e - s) of the Gram block of [s, e), in O(n^2)
    kernel evaluations and no matrix."""
    trace = 0.0
    total = 0.0
    for i in range(s, e):
        k_ii = kernel(x, i, i, kind, gamma, norms)
        trace += k_ii
        row = 0.0
        for j in range(i + 1, e):
            row += kernel(x, i, j, kind, gamma, norms)
        total += k_ii + 2.0 * row
    return trace - total / (e - s)


@njit(cache=True)
def segment_cost(x, s, e, kind, gamma, norms, buf):
    """Cost of [s, e); ``buf`` holds at least (e - s) * d values."""
    if kind == L2:
        return _l2_segment(x, s, e, buf)
    if kind == L1:
        return _l1_segment(x, s, e, buf)
    return _kernel_segment(x, s, e, kind, gamma, norms)


@njit(cache=True)
def acc_new(kind, d, capacity):
    """Statistics of a segment grown one sample at a time, for l1 and l2.

    l2: Welford's running mean and sum of squared deviations, per channel.
    l1: per channel, the lower half of the values as a max-heap (of negated
    values) and the upper half as a min-heap, with compensated sums of each.
    """
    heaps = np.empty((2, d, capacity if kind == L1 else 0))
    sizes = np.zeros((2, d), dtype=np.int64)
    # l2: mean, squared deviations; l1: lower sum, its error, upper sum, its error
    sums = np.zeros((6, d))
    return heaps, sizes, sums


@njit(cache=True)
def acc_reset(acc):
    _, sizes, sums = acc
    sizes[:] = 0
    sums[:] = 0.0


@njit(cache=True)
def _heap_push(h, size, v):
    """Push v on the min-heap h[:size]."""
    i = size
    while i > 0:
        parent = (i - 1) >> 1
        if h[parent] <= v:
            break
        h[i] = h[parent]
        i = parent
    h[i] = v


@njit(cache=True)
def _heap_pop(h, size):
    """Pop the smallest value of the min-heap h[:size]."""
    top = h[0]
    size -= 1
    last = h[size]
    i = 0
    while True:
        child = 2 * i + 1
        if child >= size:
            break
        if child + 1 < size and h[child + 1] < h[child]:
            child += 1
        if last <= h[child]:
            break
        h[i] = h[child]
        i = child
    h[i] = last
    return top


@njit(cache=True)
def _add(sums, row, c, v):
    """Neumaier's compensated summation: sums[row, c] += v, the rounding error
    accumulated in sums[row + 1, c]."""
    total = sums[row, c]
    t = total + v
    if abs(total) >= abs(v):
        sums[row + 1, c] += (total - t) + v
    else:
        sums[row + 1, c] += (v - t) + total
    sums[row, c] = t


@njit(cache=True)
def acc_push(x, i, count, kind, acc):
    """Add sample i to the segment, which then holds ``count`` samples."""
    heaps, sizes, sums = acc
    for c in range(x.shape[1]):
        v = x[i, c]
        if kind == L2:
            delta = v - sums[0, c]
            sums[0, c] += delta / count
            sums[1, c] += delta * (v - sums[0, c])
            continue
        low = heaps[0, c]
        high = heaps[1, c]
        if sizes[0, c] == 0 or v <= -low[0]:
            _heap_push(low, sizes[0, c], -v)
            sizes[0, c] += 1
            _add(sums, 2, c, v)
        else:
            _heap_push(high, sizes[1, c], v)
            sizes[1, c] += 1
            _add(sums, 4, c, v)
        if sizes[0, c] > sizes[1, c] + 1:
            t = -_heap_pop(low, sizes[0, c])
            sizes[0, c] -= 1
            _add(sums, 2, c, -t)
            _heap_push(high, sizes[1, c], t)
            sizes[1, c] += 1
            _add(sums, 4, c, t)
        elif sizes[1, c] > sizes[0, c]:
            t = _heap_pop(high, sizes[1, c])
            sizes[1, c] -= 1
            _add(sums, 4, c, -t)
            _heap_push(low, sizes[0, c], -t)
            sizes[0, c] += 1
            _add(sums, 2, c, t)


@njit(cache=True)
def acc_value(kind, acc):
    """Cost of the segment. l2: the squared deviations. l1: the absolute
    deviations from the median, i.e. the upper half's sum minus the lower
    half's, plus the median when the lower half holds it alone."""
    heaps, sizes, sums = acc
    total = 0.0
    for c in range(sums.shape[1]):
        if kind == L2:
            total += sums[1, c]
        else:
            v = (sums[4, c] + sums[5, c]) - (sums[2, c] + sums[3, c])
            if sizes[0, c] > sizes[1, c]:
                v -= heaps[0, c, 0]
            total += v
    return total


@njit(cache=True)
def kernel_extend(x, t, lo, kind, gamma, norms, diag, S):
    """Extend every segment [s, t - 1) with s >= lo by the sample t - 1: then
    ``diag[t]`` is the prefix sum of k(x_i, x_i) and ``S[s]`` the sum of
    k(x_i, x_j) over [s, t)^2 (the update of ruptures' C ekcpd solvers)."""
    k_tt = kernel(x, t - 1, t - 1, kind, gamma, norms)
    diag[t] = diag[t - 1] + k_tt
    acc = 0.0
    for s in range(t - 1, lo - 1, -1):
        acc += kernel(x, s, t - 1, kind, gamma, norms)
        S[s] += 2.0 * acc - k_tt
