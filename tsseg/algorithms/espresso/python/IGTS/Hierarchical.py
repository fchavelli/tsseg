"""Python translation of ``Hierarchical.m``."""

from __future__ import annotations

import numpy as np

from .IG_Cal import ig_cal
from ..utils.Clean_TS import clean_ts

__all__ = ["hierarchical"]


def hierarchical(integ_ts: np.ndarray, k: int, step: int, double_flag: int) -> tuple[np.ndarray, np.ndarray]:
    """Greedy hierarchical segmentation using information gain."""

    integ = clean_ts(integ_ts, double_flag)
    nu_ts, le_ts = integ.shape
    step = max(int(step), 1)

    tt = np.zeros(k + 1, dtype=int)
    tt[0] = 0
    tt[-1] = le_ts
    max_ig = np.zeros(k, dtype=float)

    for i in range(k):
        for j in range(0, le_ts, step):
            candidate_tt = tt.copy()
            candidate_tt[i] = j
            candidate_tt.sort()
            ig = ig_cal(integ, candidate_tt, i)
            if ig > max_ig[i]:
                max_ig[i] = ig
                tt = candidate_tt

    return tt[:-1], max_ig
