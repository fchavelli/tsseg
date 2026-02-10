"""Utilities for working with dynamic programming path matrices."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def from_path_matrix_to_bkps_list(
    path_matrix_flat: Sequence[int],
    n_bkps: int,
    n_samples: int,
    n_bkps_max: int,
    jump: int,
) -> list[int]:
    """Reconstruct breakpoint list from a flattened path matrix.

    The implementation mirrors the original C helper used by ruptures but is
    written in pure NumPy to avoid compiling extensions within the vendored
    copy.
    """

    if n_bkps < 0:
        raise ValueError("n_bkps must be non-negative")
    if jump <= 0:
        raise ValueError("jump must be positive")

    q = int(math.ceil(n_samples / float(jump)))
    bkps = np.empty(n_bkps + 1, dtype=int)
    bkps[-1] = q
    for k in range(1, n_bkps + 1):
        prev_index = bkps[-k]
        flat_index = prev_index * (n_bkps_max + 1) + (n_bkps - k + 1)
        bkps[-k - 1] = int(path_matrix_flat[flat_index])
    bkps *= jump
    bkps[-1] = n_samples
    return bkps.tolist()
