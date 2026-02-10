"""Python translation of ``runSegmentation.m``."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .calculateMotifDensityMatrix import calculate_motif_density_matrix
from .calculateSemanticDensityMatrix import calculate_semantic_density_matrix

__all__ = [
    "run_segmentation",
    "segment_time_series",
    "norm_cross_count_all",
]


def run_segmentation(
    matrix_profile: np.ndarray,
    mp_index: np.ndarray,
    sl_window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mirror the MATLAB segmentation helper with Python tooling."""

    crosscount1 = segment_time_series(mp_index)
    crosscount = norm_cross_count_all(crosscount1, sl_window)

    wcc1, new_wcc1 = calculate_motif_density_matrix(matrix_profile, mp_index)
    wcc = norm_cross_count_all(wcc1, sl_window)
    new_wcc = norm_cross_count_all(new_wcc1, sl_window)

    sem_arc_set, new_cc = calculate_semantic_density_matrix(matrix_profile, mp_index, 5, sl_window)
    semcc = norm_cross_count_all(sem_arc_set, sl_window)

    return crosscount, wcc, semcc, sem_arc_set, new_wcc, new_cc


def segment_time_series(mp_index: np.ndarray) -> np.ndarray:
    """Approximate the arc-crossing counts used in the MATLAB scripts.

    The original repository references a helper that is not included in the
    shipped ``code`` folder. This approximation recreates the intended behaviour
    by counting the number of arcs covering each position.
    """

    indices = np.asarray(mp_index, dtype=int).reshape(-1)
    profile_len = indices.size
    counts = np.zeros(profile_len, dtype=float)

    for idx, partner in enumerate(indices):
        if partner < 0 or partner >= profile_len:
            continue
        small = min(idx, partner)
        large = max(idx, partner)
        counts[small : large + 1] += 1.0

    return counts


def norm_cross_count_all(counts: np.ndarray, sl_window: int) -> np.ndarray:
    """Normalise counts with a simple rescaling.

    The MATLAB source relies on an external normalisation routine that is not
    provided. Here we rescale to ``[0, 1]`` to preserve relative prominence while
    documenting the assumption explicitly.
    """

    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return counts

    min_val = float(counts.min())
    max_val = float(counts.max())
    span = max_val - min_val
    if span == 0:
        return np.zeros_like(counts)
    return (counts - min_val) / span
