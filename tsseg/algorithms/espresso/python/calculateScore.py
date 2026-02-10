"""Python port of ``calculateScore.m``."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["calculate_score"]


def calculate_score(gt: np.ndarray, tt: np.ndarray, data_length: int, margin: float) -> Tuple[float, float, float, float, float]:
    """Compute precision/recall-style metrics for segmentation.

    Parameters
    ----------
    gt : np.ndarray
        Ground-truth change-point indices.
    tt : np.ndarray
        Detected change-point indices.
    data_length : int
        Length of the analysed signal.
    margin : float
        Tolerance window for counting true positives.

    Returns
    -------
    Tuple[float, float, float, float, float]
        Precision, recall, rmse1, rmse2, F-score.
    """

    gt = np.asarray(gt, dtype=float).reshape(-1)
    tt = np.asarray(tt, dtype=float).reshape(-1)

    n = gt.size
    k = tt.size
    if k == 0 or n == 0:
        return 0.0, 0.0, 1.0, 1.0, 0.0

    diff = np.abs(tt[np.newaxis, :] - gt[:, np.newaxis])
    min_v = diff.min(axis=1)
    ind = diff.argmin(axis=1)

    selected = np.zeros(k, dtype=bool)
    tp = 0
    for j, dist in enumerate(min_v):
        if dist <= margin:
            tp += 1
            selected[ind[j]] = True

    dmin_v2 = min_v.copy()
    for j, dist in enumerate(min_v):
        if dist > margin:
            deltas = np.abs(tt - gt[j])
            deltas[selected] = np.inf
            if np.isfinite(deltas).any():
                idx = int(np.argmin(deltas))
                nearest = float(deltas[idx])
                dmin_v2[j] = nearest
                selected[idx] = True

    rmse1 = float(np.sqrt((min_v**2).mean()) / data_length)
    rmse2 = float(np.sqrt((dmin_v2**2).mean()) / data_length)

    positives = np.zeros(n, dtype=int)
    detected = np.zeros(k, dtype=int)
    for j in range(n):
        hits = np.abs(tt - gt[j]) <= margin
        positives[j] = int(hits.sum())
        detected += hits.astype(int)

    fn = int((positives == 0).sum())
    fp = int((positives[positives >= 2] - 1).sum() + (detected == 0).sum())
    tp = int((positives > 0).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f_score = 0.0 if (precision == 0 and recall == 0) else 2 * (precision * recall) / (precision + recall)

    return precision, recall, rmse1, rmse2, f_score
