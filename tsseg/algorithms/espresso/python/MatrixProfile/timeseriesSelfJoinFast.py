"""Python translation of ``timeseriesSelfJoinFast.m``."""

from __future__ import annotations

import numpy as np

from numpy.random import Generator

__all__ = ["timeseries_self_join_fast"]


def timeseries_self_join_fast(
    a: np.ndarray,
    subsequence_length: int,
    *,
    rng: Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the matrix profile and index for a self-join."""

    vector = np.asarray(a, dtype=float).reshape(-1)
    n = vector.size

    if subsequence_length > n / 2:
        raise ValueError("Time series is too short relative to subsequence length")
    if subsequence_length < 4:
        raise ValueError("Subsequence length must be at least 4")

    exclusion_zone = int(round(subsequence_length / 4))
    profile_len = n - subsequence_length + 1

    matrix_profile = np.full(profile_len, np.inf, dtype=float)
    mp_index = np.zeros(profile_len, dtype=int)

    x, data_len, sumx2, sumx, meanx, sigmax2, sigmax = _fast_find_nn_pre(vector, subsequence_length)

    if isinstance(rng, Generator):
        rng_local = rng
    elif rng is None:
        rng_local = np.random.default_rng(0)
    else:
        rng_local = np.random.default_rng(rng)

    picked_idx = rng_local.permutation(profile_len)

    for idx in picked_idx:
        subsequence = vector[idx : idx + subsequence_length]
        distance_profile = _fast_find_nn(
            x,
            subsequence,
            data_len,
            subsequence_length,
            sumx2,
            sumx,
            meanx,
            sigmax2,
            sigmax,
        )
        distance_profile = np.abs(distance_profile)

        start = max(0, idx - exclusion_zone)
        end = min(profile_len, idx + exclusion_zone + 1)
        distance_profile[start:end] = np.inf

        update_pos = distance_profile < matrix_profile
        mp_index[update_pos] = idx
        matrix_profile[update_pos] = distance_profile[update_pos]

    return np.sqrt(matrix_profile), mp_index


def _fast_find_nn_pre(x: np.ndarray, m: int) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = x.size
    padded = np.pad(x, (0, n), mode="constant")
    X = np.fft.fft(padded)
    cum_sumx = np.cumsum(x)
    cum_sumx2 = np.cumsum(x**2)

    sumx2 = cum_sumx2[m - 1 : n] - np.concatenate(([0.0], cum_sumx2[: n - m]))
    sumx = cum_sumx[m - 1 : n] - np.concatenate(([0.0], cum_sumx[: n - m]))
    meanx = sumx / m
    sigmax2 = sumx2 / m - meanx**2
    sigmax = np.sqrt(np.maximum(sigmax2, 1e-12))

    return X, n, sumx2, sumx, meanx, sigmax2, sigmax


def _fast_find_nn(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    m: int,
    sumx2: np.ndarray,
    sumx: np.ndarray,
    meanx: np.ndarray,
    sigmax2: np.ndarray,
    sigmax: np.ndarray,
) -> np.ndarray:
    y = (y - y.mean()) / max(y.std(ddof=0), 1e-12)
    y = y[::-1]
    padded = np.pad(y, (0, 2 * n - y.size), mode="constant")

    Y = np.fft.fft(padded)
    Z = X * Y
    z = np.fft.ifft(Z).real

    sumy = y.sum()
    sumy2 = (y**2).sum()

    dist = (sumx2 - 2 * sumx * meanx + m * (meanx**2)) / np.maximum(sigmax2, 1e-12)
    dist -= 2 * (z[m - 1 : n] - sumy * meanx) / np.maximum(sigmax, 1e-12)
    dist += sumy2
    return np.sqrt(np.maximum(dist, 0.0))
