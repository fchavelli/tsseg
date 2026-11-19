import math
from bisect import bisect_left, insort
from collections import Counter

import numpy as np


def multivariate_l2_norm(signal: np.ndarray) -> np.ndarray:
    """
    Compute the L2 norm across dimensions for a multivariate signal.

    Parameters
    ----------
    signal : np.ndarray
        Multivariate signal of shape (n_timepoints, n_channels).

    Returns
    -------
    np.ndarray
        Univariate signal of shape (n_timepoints,) representing the L2 norm.
    """
    if signal.ndim == 1:
        return signal
    return np.linalg.norm(signal, axis=1)


def _tolerance_samples(tolerance: int | float, signal_len: int | None) -> int:
    """Tolerance in samples: a float below 1 is a fraction of ``signal_len``."""
    if isinstance(tolerance, float) and tolerance < 1.0:
        if signal_len is None:
            raise ValueError(
                "signal_len must be provided when tolerance is a fraction."
            )
        return int(tolerance * signal_len)
    return int(tolerance)


def _fuse_channel_change_points(
    per_channel: list,
    tolerance: int,
    n_cp: int | None = None,
    consensus: float = 0.0,
) -> np.ndarray:
    """Fuse the change points detected on each channel.

    Every observed change point is a candidate. A channel supports a candidate
    ``x`` if one of its change points lies at most ``tolerance`` samples from
    ``x``; the support of ``x`` is the number of supporting channels, so
    several points of one channel are a single vote. Candidates are ranked by
    support, then by strength ``sum over supporting channels of
    (tolerance + 1 - distance to the nearest point)``, then by position, and
    elected greedily: a candidate at most ``tolerance`` samples from an
    elected one is discarded. Elected indices are observed ones; points are
    never chained into clusters nor replaced by a median.

    Parameters
    ----------
    per_channel : list of array-like
        Change points detected on each channel, one entry per channel (empty
        channels included).
    tolerance : int
        Distance, in samples, at which a channel supports a candidate and two
        elected change points exclude each other. ``0`` is an exact vote.
    n_cp : int, optional
        Maximum number of change points returned. ``None`` keeps every
        elected candidate. The greedy election may return fewer than
        ``n_cp`` change points even when ``n_cp`` compatible candidates exist.
    consensus : float, default=0.0
        Fraction of the channels, in [0, 1], that must support a candidate:
        at least ``max(1, ceil(consensus * n_channels))`` channels.

    Returns
    -------
    np.ndarray
        Sorted array of change point indices (``int64``).
    """
    tolerance = int(tolerance)
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
    channels = [np.unique(np.asarray(cps, dtype=np.int64)) for cps in per_channel]
    channels_with_points = [cps for cps in channels if cps.size]
    if not channels_with_points or (n_cp is not None and n_cp <= 0):
        return np.empty(0, dtype=np.int64)

    candidates = np.unique(np.concatenate(channels_with_points))
    support = np.zeros(candidates.size, dtype=np.int64)
    strength = np.zeros(candidates.size, dtype=np.int64)
    for cps in channels_with_points:
        right = np.searchsorted(cps, candidates)
        after = cps[np.minimum(right, cps.size - 1)]
        before = cps[np.maximum(right - 1, 0)]
        distance = np.minimum(np.abs(after - candidates), np.abs(candidates - before))
        near = distance <= tolerance
        support += near
        strength += np.where(near, tolerance + 1 - distance, 0)

    needed = max(1, math.ceil(consensus * len(channels) - 1e-9))
    keep = support >= needed
    candidates, support, strength = candidates[keep], support[keep], strength[keep]

    selected: list[int] = []
    for idx in np.lexsort((candidates, -strength, -support)):
        if n_cp is not None and len(selected) >= n_cp:
            break
        x = int(candidates[idx])
        pos = bisect_left(selected, x)
        if pos < len(selected) and selected[pos] - x <= tolerance:
            continue
        if pos > 0 and x - selected[pos - 1] <= tolerance:
            continue
        insort(selected, x)
    return np.asarray(selected, dtype=np.int64)


def aggregate_change_points(
    all_cps: list[int],
    n_cp: int,
    tolerance: int | float = 0,
    signal_len: int | None = None,
) -> np.ndarray:
    """
    Aggregate change points from multiple dimensions with tolerance.

    Parameters
    ----------
    all_cps : list[int]
        List of all change point indices detected across all dimensions.
    n_cp : int
        Number of change points to return.
    tolerance : int | float, default=0
        Tolerance window for grouping change points.
        If float < 1.0, it is interpreted as a fraction of signal_len.
    signal_len : int, optional
        Length of the signal, required if tolerance is a fraction.

    Returns
    -------
    np.ndarray
        Sorted array of aggregated change point indices.
    """
    if not all_cps:
        return np.empty(0, dtype=int)

    all_cps = sorted(all_cps)

    tol_val = _tolerance_samples(tolerance, signal_len)

    if tol_val == 0:
        # Exact matching
        counts = Counter(all_cps)
        top_candidates = counts.most_common(n_cp)
        pred = sorted([idx for idx, count in top_candidates])
        return np.array(pred, dtype=int)

    # Tolerance based clustering
    clusters = []
    if all_cps:
        current_cluster = [all_cps[0]]
        for cp in all_cps[1:]:
            if cp - current_cluster[-1] <= tol_val:
                current_cluster.append(cp)
            else:
                clusters.append(current_cluster)
                current_cluster = [cp]
        clusters.append(current_cluster)

    # Score clusters by size, use median as representative
    cluster_info = []
    for cl in clusters:
        score = len(cl)
        rep = int(np.median(cl))
        cluster_info.append((score, rep))

    # Sort by score desc
    cluster_info.sort(key=lambda x: x[0], reverse=True)

    # Take top n_cp
    top_clusters = cluster_info[:n_cp]

    # Return sorted representatives
    pred = sorted([rep for score, rep in top_clusters])
    return np.array(pred, dtype=int)


def consensus_change_points(
    per_channel: list,
    tolerance: int,
    consensus: float,
) -> np.ndarray:
    """Change points detected by enough channels, without a target count.

    Every observed change point is a candidate, supported by each channel
    with a change point at most ``tolerance`` samples away. A candidate is
    kept if at least ``ceil(consensus * n_channels)`` distinct channels (one
    at least) support it, then candidates are elected greedily by support,
    strength and position, an elected change point discarding the candidates
    at most ``tolerance`` samples away. The returned indices are observed
    ones (see ``_fuse_channel_change_points``).

    Parameters
    ----------
    per_channel : list of array-like
        Change points detected on each channel, one entry per channel.
    tolerance : int
        Distance, in samples, at which a channel supports a candidate.
    consensus : float
        Fraction of the channels, in [0, 1], that must support a change point.

    Returns
    -------
    np.ndarray
        Sorted array of change point indices.
    """
    return _fuse_channel_change_points(per_channel, tolerance, consensus=consensus)


def create_state_labels(changepoints, n_timepoints):
    """
    Create segment labels from a list of changepoints.

    Each segment gets its own label (``0, 1, 2, ...`` in time order): a state
    is never reused.

    Parameters
    ----------
    changepoints : list of int
        Changepoint indices, in any order and with or without the boundaries
        ``0`` and ``n_timepoints``. Indices beyond ``n_timepoints`` are ignored.
    n_timepoints : int
        The total number of timepoints in the series.

    Returns
    -------
    np.ndarray
        A 1D integer array of segment labels of shape (n_timepoints,).
    """
    changepoints = np.array(sorted(list(set(changepoints))))

    # Ensure start and end points are included
    if 0 not in changepoints:
        changepoints = np.insert(changepoints, 0, 0)

    if n_timepoints not in changepoints:
        # use last changepoint if it's larger than n_timepoints
        if changepoints[-1] < n_timepoints:
            changepoints = np.append(changepoints, n_timepoints)

    labels = np.zeros(n_timepoints, dtype=int)

    current_label = 0
    for i in range(len(changepoints) - 1):
        start = int(changepoints[i])
        end = int(changepoints[i + 1])
        if start < n_timepoints:
            # cap end to n_timepoints
            end = min(end, n_timepoints)
            labels[start:end] = current_label
            current_label += 1

    return labels


def extract_cps(label_seq: np.ndarray) -> np.ndarray:
    """
    Extract changepoints from a sequence of state labels.

    A changepoint is defined as the first timepoint of a new state.

    Parameters
    ----------
    label_seq : np.ndarray
        A 1D array of state labels of shape (n_timepoints,).

    Returns
    -------
    np.ndarray
        A 1D array of changepoint indices.
    """
    label_diffs = label_seq[:-1] != label_seq[1:]
    return np.arange(label_seq.shape[0] - 1)[label_diffs] + 1
