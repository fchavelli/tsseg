"""Python port of ``calculateSemanticDensityMatrix.m``."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = ["calculate_semantic_density_matrix"]


def calculate_semantic_density_matrix(
    matrix_profile: np.ndarray,
    mp_index: np.ndarray,
    k: int,
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract semantic density information from matrix profile arcs.

    Parameters
    ----------
    matrix_profile : np.ndarray
        Matrix profile values for a single series.
    mp_index : np.ndarray
        Matrix profile indices (0-based) for the same series.
    k : int
        Maximum chain length to explore.
    m : int
        Window length used when computing the matrix profile.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Aggregated arc weights and a heuristic secondary spectrum mirroring the
        MATLAB behaviour.
    """

    mp = np.asarray(matrix_profile, dtype=float).reshape(-1)
    indices = np.asarray(mp_index, dtype=int).reshape(-1)
    profile_len = indices.size

    dontcare = profile_len  # sentinel index beyond the valid range
    threshold = 2.0 * float(np.nanmax(mp, initial=0.0))

    if mp.ndim != 1 or indices.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")

    arc_set: List[List[int]] = [[int(idx)] for idx in indices]
    arc_cost: List[List[float]] = [[float(cost)] for cost in mp]

    last_arc_set = indices.copy()
    last_arc_cost = mp.copy()

    for _ in range(max(k, 0)):
        arc_set, arc_cost, last_arc_set, last_arc_cost = _extract_new_arc_set(
            mp,
            indices,
            arc_set,
            arc_cost,
            threshold,
            last_arc_set,
            last_arc_cost,
            m,
            dontcare,
        )
        if not np.any(last_arc_set < dontcare):
            break

    min_entries = np.array([min(lst) if lst else idx for idx, lst in enumerate(arc_set)], dtype=float)
    max_entries = np.array([max(lst) if lst else idx for idx, lst in enumerate(arc_set)], dtype=float)

    offset = np.arange(profile_len, dtype=float)
    totmin = float(np.min(np.abs(min_entries - offset), initial=0.0))
    totmax = float(np.max(np.abs(max_entries - offset), initial=1.0))
    denom = max(totmax - totmin, 1.0)

    nnmark = np.zeros(profile_len, dtype=float)
    for j, partners in enumerate(arc_set):
        for partner in partners:
            if partner == dontcare or partner < 0 or partner >= profile_len:
                continue
            small = min(j, partner)
            large = max(j, partner)
            length = large - small
            weight = 1.0 - ((length - totmin) / denom)
            nnmark[small : large + 1] += weight

    # Secondary spectrum: reuse total partner counts as a crude proxy. The MATLAB
    # sources only returned a single value, so presenting both gives parity with
    # the call sites that expect two outputs.
    partner_counts = np.array([len([p for p in partners if p != dontcare]) for partners in arc_set], dtype=float)

    return nnmark, partner_counts


def _extract_new_arc_set(
    matrix_profile: np.ndarray,
    mp_index: np.ndarray,
    arc_set: List[List[int]],
    arc_cost: List[List[float]],
    threshold: float,
    last_arc_set: np.ndarray,
    last_arc_cost: np.ndarray,
    m: int,
    dontcare: int,
) -> Tuple[List[List[int]], List[List[float]], np.ndarray, np.ndarray]:
    profile_len = mp_index.size

    initial_arcs = last_arc_set.copy()
    mask = last_arc_cost > threshold
    initial_arcs[mask] = dontcare

    temp = np.concatenate([initial_arcs, np.array([dontcare], dtype=int)])
    new_arcs = temp[np.clip(initial_arcs, 0, dontcare)]

    temp_arc_cost = np.full(profile_len, threshold + 1.0, dtype=float)
    temp_arc_set = np.full(profile_len, dontcare, dtype=int)

    quarter_win = max(int(round(m / 4.0)), 0)

    for i in range(profile_len):
        candidate = int(new_arcs[i])
        if candidate != dontcare and (candidate > i + quarter_win or candidate < i - quarter_win):
            arc_set[i].append(candidate)
            cost = float(last_arc_cost[i] + last_arc_cost[last_arc_set[i]] if last_arc_set[i] < profile_len else threshold + 1.0)
            arc_cost[i].append(cost)
            temp_arc_cost[i] = cost
            temp_arc_set[i] = candidate
        else:
            temp_arc_cost[i] = threshold + 1.0
            temp_arc_set[i] = dontcare

    return arc_set, arc_cost, temp_arc_set, temp_arc_cost
