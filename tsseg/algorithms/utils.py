import numpy as np
from collections import Counter


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


def aggregate_change_points(all_cps: list[int], n_cp: int, tolerance: int | float = 0, signal_len: int | None = None) -> np.ndarray:
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

    # Resolve tolerance
    tol_val = 0
    if isinstance(tolerance, float) and tolerance < 1.0:
        if signal_len is None:
            raise ValueError("signal_len must be provided when tolerance is a fraction.")
        tol_val = int(tolerance * signal_len)
    else:
        tol_val = int(tolerance)
    
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


def create_state_labels(changepoints, n_timepoints):
    """
    Create state labels from a list of changepoints.

    Parameters
    ----------
    changepoints : list of int
        List of changepoint indices. The first element should be 0.
    n_timepoints : int
        The total number of timepoints in the series.

    Returns
    -------
    np.ndarray
        A 1D array of state labels of shape (n_timepoints,).
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
        end = int(changepoints[i+1])
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