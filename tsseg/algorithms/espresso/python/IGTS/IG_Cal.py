"""Python port of ``IG_Cal.m``."""

from __future__ import annotations

import numpy as np

from .Sh_Entropy import sh_entropy

__all__ = ["ig_cal"]


def ig_cal(integ_ts: np.ndarray, pos_tt1: np.ndarray, k: int) -> float:
    """Compute information gain for a proposed segmentation."""

    integ = np.asarray(integ_ts, dtype=float)
    pos_tt = np.sort(np.asarray(pos_tt1, dtype=int))

    nu_ts, le_ts = integ.shape

    ts_dist = integ[:, le_ts - 1]
    ig = sh_entropy(ts_dist)

    last_id = 0
    limit = min(k + 1, pos_tt.size)
    for idx in range(limit):
        end = pos_tt[idx]
        end = min(max(int(end), last_id + 1), le_ts)
        segment = integ[:, end - 1] - integ[:, last_id]
        ig -= ((end - last_id) / le_ts) * sh_entropy(segment)
        last_id = end

    return float(ig)
