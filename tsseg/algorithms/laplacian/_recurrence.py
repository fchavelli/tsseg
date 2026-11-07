# Adapted from librosa 0.11.0 (``librosa/segment.py``: ``recurrence_matrix``
# in affinity mode with the default bandwidth, ``recurrence_to_lag``,
# ``lag_to_recurrence`` and ``timelag_filter``).
#
# ISC License
#
# Copyright (c) 2013--2023, librosa development team.
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""Mutual k-NN recurrence affinity and diagonal median filter.

Dense re-implementations of the two librosa functions used by the Laplacian
segmentation recipe, with the same outputs (checked against librosa 0.11.0)
and without the librosa dependency:

- :func:`recurrence_affinity` is ``librosa.segment.recurrence_matrix(F.T,
  k=k, width=width, metric=metric, sym=True, mode="affinity")``;
- :func:`timelag_median_filter` is ``librosa.segment.timelag_filter(
  scipy.ndimage.median_filter)(R, size=(1, size))``, computed one lag at a
  time so that memory stays at two ``(t, t)`` arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from sklearn.neighbors import NearestNeighbors

__all__ = ["recurrence_affinity", "timelag_median_filter"]


def recurrence_affinity(
    F: np.ndarray,
    *,
    width: int,
    k: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """Symmetric mutual k-NN affinity between the rows of ``F``.

    Parameters
    ----------
    F : np.ndarray of shape (t, d)
        One feature vector per frame.
    width : int
        Links between frames closer than ``width`` in time are removed
        (``1 <= width < (t - 1) // 2``).
    k : int or None, default=None
        Neighbours kept per frame; ``None`` gives librosa's default
        ``2 * ceil(sqrt(t - 2 * width + 1))``.
    metric : str, default="cosine"
        Distance of :class:`sklearn.neighbors.NearestNeighbors`.

    Returns
    -------
    np.ndarray of shape (t, t)
        ``exp(-d_ij / b)`` where frames ``i`` and ``j`` are each among the
        other's ``k`` nearest neighbours at distance ``d_ij > 0``, 0
        elsewhere; ``b`` is the median over frames of the distance to their
        farthest kept neighbour. Exactly symmetric.

    Notes
    -----
    As in librosa, a pair at distance exactly 0 is not a link (sparse storage
    drops it) and the farthest of the ``k`` kept links is measured after the
    mutual filter.
    """
    F = np.asarray(F, dtype=float)
    t = F.shape[0]
    if width < 1 or width >= (t - 1) // 2:
        raise ValueError(
            f"width={width} must be at least 1 and below (t - 1) // 2 = {(t - 1) // 2}."
        )
    if k is None:
        k = 2 * np.ceil(np.sqrt(t - 2 * width + 1))
    k = int(k)
    knn = NearestNeighbors(
        n_neighbors=min(t - 1, k + 2 * width), metric=metric, algorithm="auto"
    )
    knn.fit(F)
    dist, ind = knn.kneighbors()  # the query frame itself is excluded

    # k nearest links of each frame, outside the band |i - j| < width. librosa
    # stores the graph sparsely, so a distance of 0 is no link, and it ranks a
    # row's links by ``np.argsort`` of their distances in column order: the
    # same sort is applied here, so that ties resolve identically.
    D = np.zeros((t, t))
    rows = np.arange(t)
    keep = (np.abs(ind - rows[:, None]) >= width) & (dist != 0)
    for i in rows:
        cols, d = ind[i][keep[i]], dist[i][keep[i]]
        order = np.argsort(cols)
        cols, d = cols[order], d[order]
        top = np.argsort(d)[:k]
        D[i, cols[top]] = d[top]

    # mutual neighbours only (``rec.minimum(rec.T)`` in librosa)
    linked = (D > 0) & (D.T > 0)
    D = np.where(linked, np.minimum(D, D.T), 0.0)

    # bandwidth: median over frames of the distance to the farthest link;
    # frames without a link do not count
    far = np.where(linked.any(axis=1), D.max(axis=1), np.nan)
    if not np.any(np.isfinite(far)):
        return np.zeros((t, t))
    bandwidth = float(np.nanmedian(far))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(linked, np.exp(D / (-1 * bandwidth)), 0.0)
    return R


def timelag_median_filter(R: np.ndarray, size: int) -> np.ndarray:
    """Median filter of size ``size`` along the diagonals of ``R``.

    librosa maps ``R`` to lag coordinates (zero padded to ``2t`` lags), filters
    each lag along time with ``scipy.ndimage.median_filter`` (``reflect``
    boundaries) and maps back. Entry ``(i + l, i)`` is filtered with the
    other entries of lag ``l`` and entry ``(i - l, i)`` with those of lag
    ``-l``; the zero padding enters near the ends of each diagonal, which is
    why the output is not symmetric even when ``R`` is.
    """
    R = np.asarray(R, dtype=float)
    t = R.shape[0]
    if size <= 1:
        return R.copy()
    out = np.empty_like(R)
    cols = np.arange(t)
    row = np.empty(t)
    for lag in range(t):  # entries (i + lag, i), i < t - lag
        i = cols[: t - lag]
        row[: t - lag] = R[i + lag, i]
        row[t - lag :] = 0.0
        out[i + lag, i] = median_filter(row, size=size, mode="reflect")[: t - lag]
    for lag in range(1, t):  # entries (i - lag, i), i >= lag
        i = cols[lag:]
        row[:lag] = 0.0
        row[lag:] = R[i - lag, i]
        out[i - lag, i] = median_filter(row, size=size, mode="reflect")[lag:]
    return out
