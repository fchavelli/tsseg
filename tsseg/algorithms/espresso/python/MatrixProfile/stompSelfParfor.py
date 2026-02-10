"""Simplified Python analogue of ``stompSelfParfor.m``."""

from __future__ import annotations

import numpy as np

from .timeseriesSelfJoinFast import timeseries_self_join_fast

__all__ = ["stomp_self_parfor"]


def stomp_self_parfor(data: np.ndarray, sub_len: int, worker_num: int = 1):
    """Fallback implementation that delegates to ``timeseries_self_join_fast``.

    The MATLAB version exploits parallel workers; in Python we rely on the FFT
    backend used by :func:`timeseries_self_join_fast` which already has
    vectorised behaviour.
    """

    return timeseries_self_join_fast(data, sub_len)
