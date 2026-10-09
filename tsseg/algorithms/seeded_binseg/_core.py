"""Interval systems, selection rules and information criterion of seeded binary
segmentation (Kovács et al., 2023) and wild binary segmentation (Fryzlewicz,
2014), written from the papers.

Intervals are 0-based and half-open: ``(start, end)`` is the segment
``x[start:end]``, the interval written ``(start, end]`` in the papers. A change
point ``b`` splits it into ``x[start:b]`` and ``x[b:end]`` (``b`` is the first
point of the new segment), and an interval *contains* ``b`` when
``start < b < end``.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

__all__ = [
    "greedy_path",
    "narrowest_over_threshold",
    "narrowest_path",
    "seeded_intervals",
    "ssic",
    "wild_intervals",
]

_TOL = 1e-9


def _ceil(x: float) -> int:
    """``ceil`` that ignores floating-point noise (``sqrt(2) ** 2`` is 2)."""
    return math.ceil(x - _TOL * max(1.0, abs(x)))


def _floor(x: float) -> int:
    """``floor`` that ignores floating-point noise."""
    return math.floor(x + _TOL * max(1.0, abs(x)))


def seeded_intervals(n: int, decay: float, min_length: int) -> np.ndarray:
    """Seeded intervals of Kovács et al. (2023, Definition 1).

    Layer ``k = 1, ..., ceil(log_{1/a} n)`` holds ``n_k = 2 ceil((1/a)^(k-1)) - 1``
    intervals of length ``l_k = n a^(k-1)`` evenly shifted by
    ``s_k = (n - l_k) / (n_k - 1)``: ``(floor((i-1) s_k), ceil((i-1) s_k + l_k)]``.

    Parameters
    ----------
    n : int
        Length of the series.
    decay : float
        Decay ``a`` in ``[1/2, 1)``.
    min_length : int
        Intervals with fewer points are dropped (``m`` in the paper).

    Returns
    -------
    np.ndarray
        ``(n_intervals, 2)`` array of ``(start, end)``, without duplicates
        (the paper notes that rounding repeats some small intervals and that
        they can be dropped), sorted.
    """
    if not 0.5 <= decay < 1.0:
        raise ValueError(f"decay must lie in [0.5, 1), got {decay}")
    if n < max(2, min_length):
        return np.empty((0, 2), dtype=np.int64)
    n_layers = _ceil(math.log(n) / math.log(1.0 / decay))
    found: set[tuple[int, int]] = set()
    for k in range(1, n_layers + 1):
        n_k = 2 * _ceil((1.0 / decay) ** (k - 1)) - 1
        l_k = n * decay ** (k - 1)
        s_k = (n - l_k) / (n_k - 1) if n_k > 1 else 0.0
        for i in range(n_k):
            start = max(0, _floor(i * s_k))
            end = min(n, _ceil(i * s_k + l_k))
            if end - start >= min_length:
                found.add((start, end))
    return np.array(sorted(found), dtype=np.int64).reshape(-1, 2)


def wild_intervals(
    n: int, n_intervals: int, min_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Random intervals of wild binary segmentation (Fryzlewicz, 2014).

    Both ends of each of the ``n_intervals`` intervals are drawn independently,
    uniformly and with replacement from ``{0, ..., n}``; draws with fewer than
    ``min_length`` points are dropped. The whole series ``(0, n)`` is always
    added (the optional augmentation of Fryzlewicz, 2014, at the top level).

    Returns
    -------
    np.ndarray
        ``(n_kept, 2)`` array of ``(start, end)``, without duplicates, sorted.
    """
    if n < max(2, min_length):
        return np.empty((0, 2), dtype=np.int64)
    ends = rng.integers(0, n + 1, size=(n_intervals, 2))
    ends.sort(axis=1)
    ends = ends[ends[:, 1] - ends[:, 0] >= min_length]
    ends = np.vstack([ends, [[0, n]]])
    return np.unique(ends, axis=0).astype(np.int64)


def _contains_any(accepted: list[int], start: int, end: int) -> bool:
    """Whether the sorted ``accepted`` has a point strictly inside (start, end)."""
    j = bisect.bisect_right(accepted, start)
    return j < len(accepted) and accepted[j] < end


def greedy_path(
    starts: np.ndarray, ends: np.ndarray, bkps: np.ndarray, gains: np.ndarray
) -> list[tuple[int, float]]:
    """Greedy selection over a fixed collection of intervals, whole path.

    The interval with the largest maximal gain gives the first change point
    (its best split); every interval that contains it is dropped; the largest
    remaining gain gives the next one, and so on (Kovács et al., 2023, Sec.
    2.4; Fryzlewicz, 2014, WBS without augmentation). Gains are thus
    non-increasing along the path, and a threshold keeps one of its prefixes.

    Parameters
    ----------
    starts, ends, bkps, gains : np.ndarray
        One entry per interval with an admissible split: bounds, best split
        and its gain.

    Returns
    -------
    list of (int, float)
        Change points and their gains, in the order they are selected.
    """
    # largest gain first; among equal gains the narrower, then the leftmost
    order = np.lexsort((starts, ends - starts, -gains))
    accepted: list[int] = []
    path: list[tuple[int, float]] = []
    for i in order:
        if not gains[i] > 0:
            break
        if _contains_any(accepted, starts[i], ends[i]):
            continue
        bisect.insort(accepted, int(bkps[i]))
        path.append((int(bkps[i]), float(gains[i])))
    return path


def _narrowness_order(starts, ends, gains) -> np.ndarray:
    """Intervals by length, then decreasing gain, then position."""
    return np.lexsort((starts, -gains, ends - starts))


def narrowest_over_threshold(
    starts: np.ndarray,
    ends: np.ndarray,
    bkps: np.ndarray,
    gains: np.ndarray,
    threshold: float,
) -> list[tuple[int, float]]:
    """Narrowest-over-threshold selection (Baranowski et al., 2019).

    Among the intervals whose maximal gain exceeds ``threshold``, the
    narrowest gives a change point (its best split); every interval that
    contains it is dropped, and the narrowest of those left gives the next one,
    until none is left.

    Returns
    -------
    list of (int, float)
        Change points and the gains that selected them, sorted by position.
    """
    order = _narrowness_order(starts, ends, gains)
    accepted: list[int] = []
    found: list[tuple[int, float]] = []
    for i in order:
        if not gains[i] > threshold:
            continue
        if _contains_any(accepted, starts[i], ends[i]):
            continue
        bisect.insort(accepted, int(bkps[i]))
        found.append((int(bkps[i]), float(gains[i])))
    return sorted(found)


def narrowest_path(
    starts: np.ndarray,
    ends: np.ndarray,
    bkps: np.ndarray,
    gains: np.ndarray,
    stop_at: int | None = None,
):
    """Narrowest-over-threshold solutions for decreasing thresholds.

    The threshold goes down through the distinct maximal gains; each step
    makes more intervals eligible. A new eligible interval changes the
    solution only if no change point selected before it (by a narrower
    interval) lies inside it; the solution is then recomputed from that
    interval on. Unlike the greedy path, the solutions are not nested and
    their size does not grow monotonically (Kovács et al., 2023, Sec. 2.4).

    Parameters
    ----------
    stop_at : int or None
        Stop after the first solution with at least this many change points.

    Yields
    ------
    list of (int, float)
        Each distinct solution, as change points and the gains that selected
        them, sorted by position.
    """
    order = _narrowness_order(starts, ends, gains)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    by_gain = np.lexsort((rank, -gains))
    eligible: list[int] = []  # ranks of the eligible intervals, sorted
    solution: list[tuple[int, int, float]] = []  # (rank, change point, gain)
    pos = 0
    while pos < len(by_gain):
        g = gains[by_gain[pos]]
        if not g > 0:
            break
        changed_from = None
        while pos < len(by_gain) and gains[by_gain[pos]] == g:
            i = by_gain[pos]
            pos += 1
            r = int(rank[i])
            bisect.insort(eligible, r)
            if changed_from is not None and r > changed_from:
                continue  # recomputed below anyway
            prefix = [cp for rk, cp, _ in solution if rk < r]
            if any(starts[i] < cp < ends[i] for cp in prefix):
                continue
            changed_from = r if changed_from is None else min(changed_from, r)
        if changed_from is None:
            continue
        kept = [item for item in solution if item[0] < changed_from]
        accepted = sorted(cp for _, cp, _ in kept)
        for r in eligible[bisect.bisect_left(eligible, changed_from) :]:
            i = order[r]
            if _contains_any(accepted, starts[i], ends[i]):
                continue
            bisect.insort(accepted, int(bkps[i]))
            kept.append((r, int(bkps[i]), float(gains[i])))
        solution = kept
        yield sorted((cp, gain) for _, cp, gain in solution)
        if stop_at is not None and len(solution) >= stop_at:
            return


def ssic(csum: np.ndarray, csum2: np.ndarray, change_points, alpha: float) -> float:
    """Strengthened Schwarz information criterion of a segmentation.

    ``n/2 * sum_j log(sigma_j^2) + k (d + 1)/2 * log(n)^alpha``, with
    ``sigma_j^2`` the residual variance of channel ``j`` around the segment
    means (maximum likelihood) and ``k`` change points. For ``d = 1`` it is
    the sSIC of Fryzlewicz (2014, eq. 4.1), ``n/2 log(sigma^2) + k
    log(n)^alpha``; for ``d > 1`` each channel has its own variance and each
    change point ``d + 1`` parameters (Baranowski et al., 2019, eq. 7).

    Parameters
    ----------
    csum, csum2 : np.ndarray
        ``(n + 1, d)`` cumulative sums of the (centred) signal and of its
        square, with a leading row of zeros.
    change_points : sequence of int
        Sorted change points, without 0 and n.
    alpha : float
        Exponent of the penalty (1 gives the Schwarz criterion).
    """
    n, d = csum.shape[0] - 1, csum.shape[1]
    bounds = np.array([0, *change_points, n])
    s1 = np.diff(csum[bounds], axis=0)
    s2 = np.diff(csum2[bounds], axis=0)
    rss = (s2 - s1**2 / np.diff(bounds)[:, None]).sum(axis=0)
    var = np.maximum(rss / n, np.finfo(float).tiny)
    k = len(bounds) - 2
    return float(n / 2 * np.log(var).sum() + k * (d + 1) / 2 * math.log(n) ** alpha)
