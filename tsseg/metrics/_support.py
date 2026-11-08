"""Scored support of a partially annotated reference.

A reference may leave samples without a usable label: missing truth, unknown
codes, or classes that are not scored. The metrics then take
``scored_intervals``, the half-open intervals ``[a, b)`` of the samples they
score, in original coordinates. The series keeps its length and the prediction
is never cut: deleting the unscored samples would join unrelated intervals and
create transitions that no annotator saw.

A change point ``i`` (the first sample of a new segment) can be judged only
when ``i - 1`` and ``i`` are both scored, so an interval ``[a, b)`` supports the
change points ``a + 1, ..., b - 1``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

__all__ = ["scored_islands"]


def scored_islands(
    labels: Sequence[int] | np.ndarray, unscored: Iterable[int] = (-1,)
) -> list[tuple[int, int]]:
    """Maximal runs of samples whose label is not in ``unscored``.

    Parameters
    ----------
    labels : array-like of shape (n_timepoints,)
        Reference label of every time point.
    unscored : iterable of int, default ``(-1,)``
        Labels that are not scored: missing truth, unknown codes, and, for
        state detection, classes such as a background.

    Returns
    -------
    list of tuple of int
        The half-open intervals ``(a, b)`` of the scored samples, sorted. Pass
        them as ``scored_intervals`` to a metric.

    Examples
    --------
    >>> scored_islands([0, 0, -1, -1, 1, 1, 2])
    [(0, 2), (4, 7)]
    """
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    scored = ~np.isin(y, np.asarray(list(unscored), dtype=y.dtype))
    edges = np.flatnonzero(np.diff(np.r_[0, scored.astype(np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _check_intervals(intervals, n_timepoints: int) -> list[tuple[int, int]]:
    """Validate ``scored_intervals`` and merge the intervals that touch.

    Raises ``ValueError`` when an interval is empty, outside ``[0, n]``, when two
    intervals overlap or are not sorted, or when no sample is scored.
    """
    arr = np.asarray(list(intervals), dtype=np.int64)
    if arr.size == 0:
        raise ValueError("scored_intervals holds no scored sample")
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("scored_intervals must be a sequence of (start, end) pairs")
    start, end = arr[:, 0], arr[:, 1]
    if (start < 0).any() or (end > n_timepoints).any() or (end <= start).any():
        raise ValueError(
            f"each scored interval [a, b) must satisfy 0 <= a < b <= {n_timepoints}"
        )
    if (start[1:] < end[:-1]).any():
        raise ValueError("scored_intervals must be sorted and must not overlap")
    merged: list[tuple[int, int]] = []
    for a, b in zip(start.tolist(), end.tolist(), strict=True):
        if merged and merged[-1][1] == a:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def _mask(islands: list[tuple[int, int]], n_timepoints: int) -> np.ndarray:
    mask = np.zeros(n_timepoints, dtype=bool)
    for a, b in islands:
        mask[a:b] = True
    return mask


def _covers_all(islands: list[tuple[int, int]], n_timepoints: int) -> bool:
    return islands == [(0, n_timepoints)]


def _boundary_distances(labels: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Distance of each point of ``labels`` to the nearest of ``anchors`` and of
    its own label changes."""
    n = len(labels)
    changes = np.flatnonzero(np.diff(labels) != 0) + 1
    marks = np.unique(np.r_[anchors, changes]).astype(np.float64)
    index = np.arange(n, dtype=np.float64)
    pos = np.searchsorted(marks, index)
    left = marks[np.clip(pos - 1, 0, len(marks) - 1)]
    right = marks[np.clip(pos, 0, len(marks) - 1)]
    return np.minimum(np.abs(index - left), np.abs(right - index))


def _island_boundary_distances(
    labels: np.ndarray, islands: list[tuple[int, int]]
) -> np.ndarray:
    """Distance of each scored point to the nearest true change of its island,
    the edges of the island counting as anchors; ``nan`` outside the islands."""
    labels = np.asarray(labels)
    out = np.full(len(labels), np.nan)
    for a, b in islands:
        out[a:b] = _boundary_distances(labels[a:b], np.array([0, b - a]))
    return out


def _supported_points(points: Iterable[int], islands: list[tuple[int, int]]):
    """The change points of each island that its support can judge."""
    p = np.asarray(sorted(int(x) for x in points), dtype=np.int64)
    out = []
    for a, b in islands:
        lo, hi = np.searchsorted(p, [a + 1, b])
        out.append(p[lo:hi].tolist())
    return out


def _directional_covering(source_segments, target_segments) -> float:
    """Length-weighted best IoU of each source segment with the target ones."""
    source_segments = list(source_segments)
    target_segments = list(target_segments)

    if not source_segments:
        return 1.0 if not target_segments else 0.0

    total_length = sum(max(0, end - start) for start, end in source_segments)
    if total_length <= 0:
        return 0.0

    if not target_segments:
        return 0.0

    # Ensure target segments are sorted for the sweep.
    target_segments.sort(key=lambda seg: seg[0])

    coverage_sum = 0.0
    target_len = len(target_segments)
    pointer = 0

    for src_start, src_end in source_segments:
        if src_end <= src_start:
            continue

        max_iou = 0.0

        while pointer < target_len and target_segments[pointer][1] <= src_start:
            pointer += 1

        idx = pointer
        while idx < target_len and target_segments[idx][0] < src_end:
            tgt_start, tgt_end = target_segments[idx]
            intersection = max(0, min(src_end, tgt_end) - max(src_start, tgt_start))
            if intersection > 0:
                union = (src_end - src_start) + (tgt_end - tgt_start) - intersection
                if union > 0:
                    max_iou = max(max_iou, intersection / union)
            idx += 1

        coverage_sum += (src_end - src_start) * max_iou

    return coverage_sum / total_length


def _masked_covering(true_points, pred_points, islands):
    """Covering in both directions, inside each scored interval of at least two
    points, weighted by the length of the interval.

    Returns the ground-truth covering, the prediction covering and the scored
    duration. Raises ``ValueError`` when no interval holds two points.
    """
    kept = [(a, b) for a, b in islands if b - a >= 2]
    if not kept:
        raise ValueError(
            "scored_intervals supports no change-point position "
            "(no interval of at least two points)"
        )
    true_groups = _supported_points(true_points, kept)
    pred_groups = _supported_points(pred_points, kept)
    gt_sum = pred_sum = 0.0
    duration = 0
    for (a, b), t, p in zip(kept, true_groups, pred_groups, strict=True):
        gt = _segments([a, *t, b])
        pred = _segments([a, *p, b])
        gt_sum += (b - a) * _directional_covering(gt, pred)
        pred_sum += (b - a) * _directional_covering(pred, gt)
        duration += b - a
    return gt_sum / duration, pred_sum / duration, duration


def _segments(edges: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def _n_supported(islands: list[tuple[int, int]]) -> int:
    """Number of change-point positions that the intervals can judge."""
    return sum(max(0, b - a - 1) for a, b in islands)
