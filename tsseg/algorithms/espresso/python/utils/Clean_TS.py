"""Python translation of ``Clean_TS.m``."""

from __future__ import annotations

import numpy as np

__all__ = ["clean_ts"]


def clean_ts(o_integ_ts: np.ndarray, double_flag: int) -> np.ndarray:
    """Normalise and optionally mirror the supplied time series."""

    integ_ts = np.asarray(o_integ_ts, dtype=float).copy()
    m, n = integ_ts.shape if integ_ts.ndim == 2 else (1, integ_ts.size)
    integ_ts = integ_ts.reshape(m, n)

    rows = [integ_ts[i].copy() for i in range(m)]

    originals = []
    mirrored_rows = []
    for row in rows:
        min_val = 0.0 if double_flag == 2 else float(row.min())
        row = row - min_val
        if double_flag != 2:
            sum_val = row.sum() / 1000.0 if row.sum() != 0 else 1.0
            row = row / sum_val
        originals.append(row)
        if double_flag == 1:
            max_val = float(row.max())
            mirrored = max_val - row
            sum_val = mirrored.sum() / 1000.0 if mirrored.sum() != 0 else 1.0
            mirrored_rows.append(mirrored / sum_val)

    # MATLAB layout: [row1, row2, ..., mirror1, mirror2, ...]
    result = np.vstack(originals + mirrored_rows)
    return np.cumsum(result, axis=1)
