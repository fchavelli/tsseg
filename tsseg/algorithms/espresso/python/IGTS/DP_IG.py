"""Python translation of ``DP_IG.m`` using numpy primitives."""

from __future__ import annotations

import numpy as np

from .Sh_Entropy import sh_entropy
from ..utils.Clean_TS import clean_ts

__all__ = ["dp_ig"]


def dp_ig(integ_ts: np.ndarray, k: int, step: int, double_flag: int) -> tuple[np.ndarray, float]:
    """Dynamic-programming segmentation guided by information gain."""

    integ = clean_ts(integ_ts, double_flag)
    nu_ts, le_ts = integ.shape
    step = max(int(step), 1)

    cost = np.full((le_ts, le_ts, k + 1), np.inf)
    cost[:, :, 0] = 0.0

    for i in range(0, le_ts, step):
        for j in range(i + 1, le_ts, step):
            ts_dist = integ[:, j] - integ[:, i]
            entropy = sh_entropy(ts_dist)
            segment_cost = ((j - i) / le_ts) * entropy
            cost[i : i + step, j : j + step, 0] = segment_cost

    pos = np.zeros((le_ts, k + 1), dtype=int)
    for b in range(1, k + 1):
        for i in range(1, le_ts):
            best = cost[0, i, b - 1]
            best_pos = 0
            for j in range(step, i, step):
                candidate = cost[0, j, b - 1] + cost[j, i, 0]
                if candidate <= best:
                    best = candidate
                    best_pos = j
            cost[0, i, b] = best
            pos[i, b] = best_pos

    max_var = float(cost[0, le_ts - 1, k])

    tt = np.zeros(k, dtype=int)
    idx = le_ts - 1
    for b in range(k, 0, -1):
        tt[b - 1] = pos[idx, b]
        idx = tt[b - 1]

    return tt, max_var
