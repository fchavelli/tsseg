import os
import shutil

import numpy as np
from numba import njit

"""
ClaSP utilities
"""

def check_input_time_series(time_series):
    """
    Check that the input time series is a 1 or 2-dimensional numpy array of numbers.

    Parameters
    ----------
    time_series : array-like
        The input time series.

    Returns
    -------
    time_series : ndarray
        The input time series as a 2-dimensional numpy array of floats or integers.

    Raises
    ------
    TypeError
        If the input time series is not an array-like object or not a 1 or 2-dimensional array.
    ValueError
        If the input time series is not composed of numbers.
    """
    if not isinstance(time_series, np.ndarray):
        raise TypeError("Input time series must be a numpy array.")

    if len(time_series.shape) not in (1, 2):
        raise ValueError("Input time series must be one or two-dimensional.")

    if not np.issubdtype(time_series.dtype, np.number):
        raise TypeError("Input time series must contain numeric values.")

    if time_series.ndim == 1:
        # make ts multi-dimensional
        return time_series.reshape(-1, 1)

    return time_series


def check_excl_radius(k_neighbours, excl_radius):
    """
    Check if the exclusion radius is larger than the amount of neighbours used for ClaSP.
    This is conceptually useful, as the excl_radius defines the min_seg_size, which
    should include at least k repetitions of a temporal pattern and itself.

    Parameters
    ----------
    k_neighbours : int
        The number of neighbours used for ClaSP.
    excl_radius : float
        The exclusion radius used for ClaSP.

    Raises
    ------
    ValueError
        If the exclusion radius is smaller than the number of neighbours used.
    """
    if excl_radius <= k_neighbours:
        raise ValueError("Exclusion radius must be larger than the number of neighbours used.")


def numba_cache_safe(func, *args, **kwargs):
    """
    Execute a function safely, handling potential issues with numba's cache. If a
    ReferenceError is caught, indicating a potential issue with the cache, it
    is cleared and the function is attempted to be executed again.

    Parameters
    ----------
    func : callable
        The function to be executed. This function can be any callable but is
        intended for use with functions that use Numba's JIT compilation.
    *args : tuple
        Positional arguments to be passed to the function.
    **kwargs : dict
        Keyword arguments to be passed to the function.

    Returns
    -------
    Any
        The return value of the `func` function.

    Raises
    ------
    ReferenceError
        If the ReferenceError persists even after clearing the Numba cache.
    """
    try:
        return func(*args, **kwargs)
    except ReferenceError as e:
        pycache_path = os.path.join(os.path.dirname(__file__), "__pycache__")

        # delete cache and try again
        if os.path.exists(pycache_path):
            shutil.rmtree(pycache_path)
            return func(*args, **kwargs)

        # other issue
        raise e


@njit(fastmath=True, cache=True)
def roll_array(arr, num, fill_value=0):
    """
    Rolls the elements of a 1D array to the right by a specified number of positions.

    This function performs a right circular shift of the input array by `num` positions. The shifted-in
    position is filled with a specified value instead of wrapping around.

    Parameters
    ----------
    arr : np.ndarray
        The 1D input array to be rolled.
    num : int
        Number of positions to roll the array to the right.
    fill_value : scalar, optional
        The value to insert at the beginning of the array after the shift. Default is 0.

    Returns
    -------
    result : np.ndarray
        The rolled array with the first `num` elements filled by `fill_value` and the rest shifted accordingly.
    """
    result = np.empty_like(arr)

    result[num] = fill_value
    result[:num] = arr[-num:]

    return result

# Normalizes multivariate time series
def normalize_time_series(ts):
    flatten = False

    if ts.ndim == 1:
        ts = ts.reshape(-1, 1)
        flatten = True

    for dim in range(ts.shape[1]):
        channel = ts[:, dim]

        # Min-max normalize channel
        try:
            channel = np.true_divide(channel - channel.min(), channel.max() - channel.min())
        except FloatingPointError:
            pass

        # Interpolate (if missing values are present)
        channel[np.isinf(channel)] = np.nan
        channel = pd.Series(channel).interpolate(limit_direction="both").to_numpy()

        # There are series that still contain NaN values
        channel[np.isnan(channel)] = 0

        ts[:, dim] = channel

    if flatten:
        ts = ts.flatten()

    return ts

"""
CLaP utilities
"""

# Create vector of state labels that map to data points
@njit(fastmath=True, cache=True)
def create_state_labels(cps, labels, ts_len):
    seg_labels = np.zeros(shape=ts_len, dtype=np.int64)

    segments = np.concatenate((
        np.array([0]),
        cps,
        np.array([ts_len])
    ))

    for idx in range(1, len(segments)):
        seg_start, seg_end = segments[idx - 1], segments[idx]
        seg_labels[seg_start:seg_end] = labels[idx - 1]

    return seg_labels


# Creates a sliding window from a time series
def create_sliding_window(time_series, window_size, stride=1):
    X = []

    for idx in range(0, time_series.shape[0], stride):
        if idx + window_size <= time_series.shape[0]:
            X.append(time_series[idx:idx + window_size])

    return np.array(X, dtype=time_series.dtype)


# Expands a label sequence from a sliding window
def expand_label_sequence(labels, window_size, stride):
    X = []

    for label in labels:
        X.extend([label] * (window_size - (window_size - stride)))

    return np.array(X, dtype=labels.dtype)


# Collapses a label sequence to its dense representation
def collapse_label_sequence(label_seq):
    labels = []

    for idx in range(1, len(label_seq)):
        if label_seq[idx - 1] != label_seq[idx]:
            labels.append(label_seq[idx - 1])

        if idx == len(label_seq) - 1:
            labels.append(label_seq[idx])

    return np.array(labels)


# Extracts CPs from a label sequence
def extract_cps(label_seq):
    label_diffs = label_seq[:-1] != label_seq[1:]
    return np.arange(label_seq.shape[0] - 1)[label_diffs] + 1