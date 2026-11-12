"""Joint MAP decoding for destination-by-source transition matrices."""

from __future__ import annotations

import numpy as np


def viterbi(log_transition, log_emission, log_initial):
    """Return the most likely state path; transition matrix zero is unused."""
    n, k = log_emission.shape
    back = np.empty((n, k), dtype=np.intp)
    score = log_initial + log_emission[0]
    for t in range(1, n):
        candidates = log_transition[t] + score[None, :]
        back[t] = np.argmax(candidates, axis=1)
        score = log_emission[t] + np.max(candidates, axis=1)
        score -= np.max(score)
    labels = np.empty(n, dtype=np.int64)
    labels[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        labels[t - 1] = back[t, labels[t]]
    return labels
