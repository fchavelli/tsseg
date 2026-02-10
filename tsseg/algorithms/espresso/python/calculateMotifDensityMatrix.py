"""Python port of ``calculateMotifDensityMatrix.m``."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["calculate_motif_density_matrix"]


def calculate_motif_density_matrix(matrix_profile: np.ndarray, mp_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate weighted motif arcs into density counts.

    Parameters
    ----------
    matrix_profile : np.ndarray
        Matrix profile values for a single time series.
    mp_index : np.ndarray
        Matrix profile indices (0-based) for each position.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``(crosscount, newc)`` matching the MATLAB outputs.
    """

    mp = np.asarray(matrix_profile, dtype=float).reshape(-1)
    indices = np.asarray(mp_index, dtype=int).reshape(-1)
    profile_len = indices.size

    nnmark = np.zeros(profile_len, dtype=float)
    newmark = np.zeros(profile_len, dtype=float)

    arc_lengths = np.abs(np.arange(profile_len) - indices)
    max_len = float(arc_lengths.max(initial=0))
    min_len = float(arc_lengths.min(initial=0))
    diff = max(max_len - min_len, 1.0)

    mean_val = float(np.nanmean(mp))
    sim_diff = np.max(np.abs(np.asarray([mp.min(initial=mean_val), mp.max(initial=mean_val)]) - mean_val))
    if sim_diff == 0:
        sim_diff = 1.0

    for idx in range(profile_len):
        partner = indices[idx]
        if partner < 0 or partner >= profile_len:
            continue
        small = min(idx, partner)
        large = max(idx, partner)
        arc_length = abs(idx - partner)
        arc_weight = 1.0 - ((arc_length - min_len) / diff) ** 2
        nnmark[small : large + 1] += arc_weight
        newmark[small : large + 1] += (1.0 + ((mp[idx] - mean_val) / sim_diff) * arc_weight)

    return nnmark, newmark
