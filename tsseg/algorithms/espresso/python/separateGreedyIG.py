"""Python translation of ``separateGreedyIG.m``."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .IGTS.IG_Cal import ig_cal
from .utils.Clean_TS import clean_ts

try:  # Prefer SciPy when available for robust peak detection
    from scipy.signal import find_peaks, peak_prominences  # type: ignore
except Exception:  # pragma: no cover - SciPy optional dependency
    find_peaks = None
    peak_prominences = None

__all__ = ["separate_greedy_ig"]


def separate_greedy_ig(
    ts: np.ndarray,
    num_segms: int,
    cc: np.ndarray,
    pdist_fraction: float,
) -> Tuple[np.ndarray, float]:
    """Greedy IG-based segmentation mirroring the MATLAB logic."""

    ts = np.asarray(ts, dtype=float)
    if ts.ndim == 1:
        ts = ts[np.newaxis, :]

    number_of_ts, data_length = ts.shape
    integ_ts = clean_ts(ts, double_flag=1)

    pdist = max(int(np.floor(data_length * pdist_fraction)), 1)
    cc_smooth = -1.0 * _gaussian_smooth(cc, window=5)

    best_tt = np.array([], dtype=int)
    best_ig = 0.0

    for d in range(number_of_ts):
        peaks, prominences = _find_peaks(cc_smooth[d], min_peak_distance=pdist)
        if peaks.size == 0:
            continue

        order = np.argsort(prominences)[::-1]
        remain_locs = peaks[order].tolist()
        tt: list[int] = []
        max_ig = np.zeros(len(remain_locs), dtype=float)

        for i in range(len(remain_locs)):
            temp_tt = None
            for candidate in list(remain_locs):
                c = np.array(tt + [candidate, data_length], dtype=int)
                ig = ig_cal(integ_ts, c, i)
                if ig > max_ig[i]:
                    max_ig[i] = ig
                    temp_tt = candidate
            if temp_tt is None:
                continue
            tt.append(temp_tt)
            remain_locs.remove(temp_tt)

        if not tt:
            continue

        t = min(num_segms - 1, len(max_ig))
        if t <= 0:
            continue

        if max_ig[t - 1] > best_ig:
            best_ig = float(max_ig[t - 1])
            best_tt = np.array(sorted(tt[:t]), dtype=int)

    return best_tt, best_ig


def _gaussian_smooth(arr: np.ndarray, window: int) -> np.ndarray:
    """Smooth ``arr`` along the last axis using a Gaussian kernel."""

    arr = np.asarray(arr, dtype=float)
    if window <= 1:
        return arr.copy()

    radius = window // 2
    sigma = max(window / 6.0, 1e-6)
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(kernel_x**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    def _convolve(vector: np.ndarray) -> np.ndarray:
        return np.convolve(vector, kernel, mode="same")

    flattened = arr.reshape(-1, arr.shape[-1])
    smoothed = np.vstack([_convolve(row) for row in flattened])
    return smoothed.reshape(arr.shape)


def _find_peaks(signal: np.ndarray, min_peak_distance: int) -> Tuple[np.ndarray, np.ndarray]:
    """Locate peaks and prominences with a SciPy-powered or fallback approach."""

    signal = np.asarray(signal, dtype=float)

    if find_peaks is not None:
        peaks, _ = find_peaks(signal, distance=min_peak_distance)
        if peaks.size == 0:
            return peaks, np.array([], dtype=float)
        prominences = peak_prominences(signal, peaks)[0]
        return peaks, np.asarray(prominences, dtype=float)

    # Fallback: detect local maxima and estimate prominence heuristically.
    peaks: list[int] = []
    prominences: list[float] = []
    length = signal.size
    for idx in range(1, length - 1):
        if signal[idx] <= signal[idx - 1] or signal[idx] <= signal[idx + 1]:
            continue
        if peaks and idx - peaks[-1] < min_peak_distance:
            if signal[idx] > signal[peaks[-1]]:
                peaks[-1] = idx
            continue
        peaks.append(idx)
        left = signal[: idx + 1]
        right = signal[idx:]
        prominences.append(float(signal[idx] - min(left.min(), right.min())))

    return np.asarray(peaks, dtype=int), np.asarray(prominences, dtype=float)
