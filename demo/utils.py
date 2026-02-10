"""Helper utilities for the demo app."""

from __future__ import annotations

import ast
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def states_to_change_points(states: Sequence[int]) -> list[int]:
    """Derive change points from state labels."""

    diffs = np.diff(states)
    cps = list(np.nonzero(diffs)[0] + 1)
    return cps


def change_points_to_states(change_points: Sequence[int], length: int) -> np.ndarray:
    """Convert a sorted change-point list to per-sample state labels."""

    labels = np.zeros(length, dtype=int)
    boundaries = [0, *sorted(cp for cp in change_points if 0 < cp < length), length]
    for idx, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        labels[start:stop] = idx
    return labels


def one_hot_from_change_points(change_points: Sequence[int], length: int) -> np.ndarray:
    vec = np.zeros(length, dtype=int)
    for cp in change_points:
        if 0 <= cp < length:
            vec[cp] = 1
    return vec


def align_states_via_hungarian(true_states: Sequence[int], pred_states: Sequence[int]) -> np.ndarray:
    """Relabel predictions to maximise agreement with ground truth."""

    true_states = np.asarray(true_states, dtype=int)
    pred_states = np.asarray(pred_states, dtype=int)
    n_true = true_states.max() + 1
    n_pred = pred_states.max() + 1
    size = max(n_true, n_pred)
    cost = np.zeros((size, size), dtype=int)
    for t, p in zip(true_states, pred_states):
        cost[t, p] -= 1  # negative for maximisation
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    remapped = np.array([mapping.get(label, label) for label in pred_states], dtype=int)
    return remapped


def parse_user_value(raw: str, default: Any) -> Any:
    """Parse a string input using ``ast.literal_eval`` with fallbacks."""

    if raw.strip() == "":
        return default
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """Apply z-normalisation along time for each dimension."""

    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (signal - mean) / std
