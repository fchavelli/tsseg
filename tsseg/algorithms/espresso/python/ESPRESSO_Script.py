"""Python translation of ``ESPRESSO_Script.m``.

This module mirrors the MATLAB entry point for ESPRESSO segmentation using the
translated helper routines contained in this package.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .MatrixProfile.timeseriesSelfJoinFast import timeseries_self_join_fast
from .calculateSemanticDensityMatrix import calculate_semantic_density_matrix
from .separateGreedyIG import separate_greedy_ig

__all__ = ["espresso", "compute_mp"]


def espresso(
    k: int,
    data: np.ndarray,
    subsequence: int,
    chain_len: int,
    *,
    pdist_fraction: float = 0.01,
    rng: np.random.Generator | int | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Run ESPRESSO segmentation on the supplied data matrix."""

    series = np.asarray(data, dtype=float)
    if series.ndim == 1:
        series = series[np.newaxis, :]
    if series.ndim != 2:
        raise ValueError("`data` must be 1D or 2D array-like")

    if rng is not None and random_state is not None:
        raise ValueError("Specify either `rng` or `random_state`, not both")

    if isinstance(rng, np.random.Generator):
        rng_local = rng
    elif rng is None:
        rng_local = np.random.default_rng(random_state)
    else:
        rng_local = np.random.default_rng(rng)

    mp, mpi = compute_mp(series, subsequence, rng=rng_local)

    wcac = np.zeros_like(mp)
    for row in range(mp.shape[0]):
        _, wcac[row] = calculate_semantic_density_matrix(mp[row], mpi[row], chain_len, subsequence)

    best_tt, _ = separate_greedy_ig(series, k, wcac, pdist_fraction)
    change_points = np.asarray(best_tt, dtype=int)
    if change_points.size:
        change_points = np.unique(np.clip(change_points, 0, series.shape[1]))
    return change_points


def compute_mp(
    integrated_ts: np.ndarray,
    subsequence: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute matrix profiles for each time series in ``integrated_ts``."""

    integrated_ts = np.asarray(integrated_ts, dtype=float)
    if integrated_ts.ndim == 1:
        integrated_ts = integrated_ts[np.newaxis, :]

    n_series, series_length = integrated_ts.shape

    if isinstance(rng, np.random.Generator):
        rng_local = rng
    else:
        rng_local = np.random.default_rng(rng)

    profile_length = series_length - subsequence + 1
    mp = np.zeros((n_series, profile_length), dtype=float)
    mpi = np.zeros((n_series, profile_length), dtype=int)

    for idx in range(n_series):
        mp[idx], mpi[idx] = timeseries_self_join_fast(
            integrated_ts[idx],
            subsequence,
            rng=rng_local,
        )

    return mp, mpi
