import warnings

import numpy as np
from scipy.signal import peak_prominences


def distance(data, window_size):
    """
    Calculates distance (dissimilarity measure) between features

    Args:
        data: array of of learned features of size (nr. of windows) x (number of shared features)
        window_size: window size used for CPD

    Returns:
        Array of dissimilarities of size ((nr. of windows)-stride)
    """

    nr_windows = np.shape(data)[0]

    index_1 = range(window_size, nr_windows, 1)
    index_2 = range(0, nr_windows - window_size, 1)

    return np.sqrt(np.sum(np.square(data[index_1] - data[index_2]), 1))


def ts_to_windows(ts, onset, window_size, stride, normalization="timeseries"):
    """Transforms time series into list of windows"""
    windows = []
    len_ts = len(ts)
    onsets = range(onset, len_ts - window_size + 1, stride)

    if normalization == "timeseries":
        for timestamp in onsets:
            windows.append(ts[timestamp : timestamp + window_size])
    elif normalization == "window":
        for timestamp in onsets:
            windows.append(
                np.array(ts[timestamp : timestamp + window_size])
                - np.mean(ts[timestamp : timestamp + window_size])
            )

    return np.array(windows)


def combine_ts(list_of_windows):
    """
    Combines a list of windows from multiple views to one list of windows

    Args:
        list_of_windows: list of windows from multiple views

    Returns:
        one array with the concatenated windows
    """
    nr_ts, nr_windows, window_size = np.shape(list_of_windows)
    tss = np.array(list_of_windows)
    new_ts = []
    for i in range(nr_windows):
        new_ts.append(tss[:, i, :].flatten())
    return np.array(new_ts)


def new_peak_prominences(distances):
    """
    Adapted calculation of prominence of peaks, based on the original scipy code

    Args:
        distances: dissimarity scores
    Returns:
        prominence scores
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        all_peak_prom = peak_prominences(distances, range(len(distances)))
    return all_peak_prom


def matched_filter(signal, window_size):
    """
    Matched filter for dissimilarity measure smoothing (and zero-delay weighted moving average filter for shared feature smoothing)

    Args:
        signal: input signal
        window_size: window size used for CPD
    Returns:
        filtered signal
    """
    mask = np.ones((2 * window_size + 1,))
    for i in range(window_size):
        mask[i] = i / (window_size**2)
        mask[-(i + 1)] = i / (window_size**2)
    mask[window_size] = window_size / (window_size**2)

    signal_out = np.zeros(np.shape(signal))

    if len(np.shape(signal)) > 1:
        for i in range(np.shape(signal)[1]):
            signal_extended = np.concatenate(
                (
                    signal[0, i] * np.ones(window_size),
                    signal[:, i],
                    signal[-1, i] * np.ones(window_size),
                )
            )
            signal_out[:, i] = np.convolve(signal_extended, mask, "valid")
    else:
        signal = np.concatenate(
            (
                signal[0] * np.ones(window_size),
                signal,
                signal[-1] * np.ones(window_size),
            )
        )
        signal_out = np.convolve(signal, mask, "valid")

    return signal_out


def calc_fft(windows, nfft=30, norm_mode="timeseries"):
    """
    Calculates the DFT for each window and transforms its length

    Args:
        windows: time series windows
        nfft: number of points used for the calculation of the DFT
        norm_mode: ensure that the timeseries / each window has zero mean

    Returns:
        frequency domain windows, each window having size nfft//2 (+1 for timeseries normalization)
    """
    mean_per_segment = np.mean(windows, axis=-1)
    mean_all = np.mean(mean_per_segment, axis=0)

    if norm_mode == "window":
        windows = windows - mean_per_segment[:, None]
        windows_fft = abs(np.fft.fft(windows, nfft))[..., 1 : nfft // 2 + 1]
    elif norm_mode == "timeseries":
        windows = windows - mean_all
        windows_fft = abs(np.fft.fft(windows, nfft))[..., : nfft // 2 + 1]

    fft_max = np.amax(windows_fft)
    fft_min = np.amin(windows_fft)

    windows_fft = 2 * (windows_fft - fft_min) / (fft_max - fft_min) - 1

    return windows_fft
