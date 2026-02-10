"""Python translation of ``calcScore.m``.

The functions keep MATLAB's naming conventions so other translated modules can
import them without additional glue code.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["calc_score"]


def calc_score(ground_truth: np.ndarray, detected_seg_loc: np.ndarray, data_length: int) -> Tuple[float, float]:
    """Compute the average and root-sum deviation between boundaries.

    Parameters
    ----------
    ground_truth : np.ndarray
        Array containing ground-truth change-point indices.
    detected_seg_loc : np.ndarray
        Array containing detected change-point indices.
    data_length : int
        Length of the time series (used for normalisation).

    Returns
    -------
    Tuple[float, float]
        (score, score2) pair mirroring the MATLAB implementation.
    """

    gt = np.asarray(ground_truth).reshape(-1)
    detected = np.asarray(detected_seg_loc).reshape(-1)

    if detected.size == 0 or gt.size == 0:
        return float("inf"), float("inf")

    diff = np.abs(detected[np.newaxis, :] - gt[:, np.newaxis])
    min_dist = diff.min(axis=1)
    sum_of_diff = float(min_dist.sum())

    score = sum_of_diff / (data_length * len(min_dist))
    score2 = float(np.sqrt(sum_of_diff**2 / data_length))
    return score, score2
