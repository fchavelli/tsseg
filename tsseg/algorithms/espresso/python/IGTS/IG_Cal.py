"""Python port of ``IG_Cal.m``."""

from __future__ import annotations

import numpy as np

from .Sh_Entropy import sh_entropy

__all__ = ["ig_cal"]


def ig_cal(integ_ts: np.ndarray, pos_tt1: np.ndarray, k: int) -> float:
    """Compute information gain for a proposed segmentation.

    Mirrors the MATLAB ``IG_Cal.m`` logic.  ``integ_ts`` is the cumulative
    sum produced by ``Clean_TS`` (0-based columns).  ``pos_tt1`` contains
    candidate split positions **and** the total length as last entry,
    matching the MATLAB convention ``c = [TT, dataLength]``.
    """

    integ = np.asarray(integ_ts, dtype=float)
    pos_tt = np.sort(np.asarray(pos_tt1, dtype=int))

    nu_ts, le_ts = integ.shape

    # Total entropy over the whole series  (MATLAB: Integ_TS(:, Le_TS))
    ts_dist = integ[:, le_ts - 1]
    ig = sh_entropy(ts_dist)

    # Walk through every segment boundary in pos_tt (including dataLength).
    # MATLAB is 1-based with last_id starting at 1; in 0-based cumsum space
    # the segment [a, b) has distribution integ[:, b-1] - integ[:, a-1],
    # with the convention integ[:, -1] == 0 for the first segment.
    last_id = 0  # corresponds to MATLAB last_id = 1
    for idx in range(pos_tt.size):
        end = int(pos_tt[idx])
        end = min(max(end, last_id + 1), le_ts)
        length = end - last_id
        if last_id == 0:
            segment = integ[:, end - 1]
        else:
            segment = integ[:, end - 1] - integ[:, last_id - 1]
        ig -= (length / le_ts) * sh_entropy(segment)
        last_id = end

    return float(ig)
