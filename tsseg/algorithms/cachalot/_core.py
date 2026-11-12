"""Numerical routines for the Cachalot change point method."""

from __future__ import annotations

import numpy as np

__all__ = [
    "lasso_candidate_points",
    "reduced_dynamic_programming",
    "select_change_count",
]


def lasso_candidate_points(y: np.ndarray, max_candidates: int) -> np.ndarray:
    """Return the first step variables entering the centered TV-Lasso path.

    For split positions ``p``, the centered design has Gram matrix
    ``G[i, j] = min(p[i], p[j]) - p[i] * p[j] / n``. Its active inverse
    is tridiagonal: applying it to the active signs only needs adjacent
    spacings. Multiplying back by *all* columns is linear interpolation
    between the active signs, with zero boundary values at 0 and n.

    This gives O(n K) time and O(n + K) working memory for K entries,
    without a dense design, Gram matrix, or linear system. Active directions
    have their coefficient's sign (or are zero), so this unweighted step
    design has no Lasso dropouts. Exact ties enter in increasing index order;
    the candidate budget can truncate a tie. Columns are not standardized.
    """
    y = np.asarray(y, dtype=float)
    n_samples = y.size
    limit = min(max_candidates, n_samples - 1)
    if limit <= 0:
        return np.empty(0, dtype=np.int64)

    # Subtract a represented observation first: unlike y - mean(y), this
    # maps *every* constant signal exactly to zero, including decimal levels.
    centered = y - y[0]
    centered -= centered.mean()
    amplitude = float(np.max(np.abs(centered)))
    if amplitude == 0:
        return np.empty(0, dtype=np.int64)
    centered /= amplitude
    correlations = -np.cumsum(centered)[:-1]
    positions = np.arange(1, n_samples)
    active = np.zeros(n_samples - 1, dtype=bool)
    entered: list[int] = []
    signs = np.zeros(n_samples - 1)
    first = int(np.argmax(np.abs(correlations)))
    correlation = float(abs(correlations[first]))
    # Scale-relative roundoff bound for the cumulative sums. Normalizing
    # first makes endpoint decisions independent of the measurement unit.
    tolerance = 16 * np.finfo(float).eps * n_samples

    while len(entered) < limit and correlation > tolerance:
        active[first] = True
        entered.append(first + 1)
        signs[first] = np.sign(correlations[first])
        if len(entered) == limit:
            break

        knots = positions[active]
        active_signs = signs[active]
        boundaries = np.concatenate(([0], knots, [n_samples]))
        padded_signs = np.concatenate(([0.0], active_signs, [0.0]))
        # G @ inv(G_A) @ signs, without the equiangular normalization:
        # the path is parameterized by the decrease in residual correlation.
        direction = np.interp(positions, boundaries, padded_signs)

        steps = np.full(n_samples - 1, np.inf)
        for sign in (1.0, -1.0):
            denominator = 1.0 - sign * direction
            valid = (~active) & (denominator > 16 * np.finfo(float).eps)
            trial = np.full(n_samples - 1, np.inf)
            trial[valid] = (correlation - sign * correlations[valid]) / denominator[
                valid
            ]
            # A negative step within roundoff is an exact tie.
            valid &= trial >= -tolerance
            steps[valid] = np.minimum(steps[valid], np.maximum(trial[valid], 0.0))

        first = int(np.argmin(steps))
        step = float(steps[first])
        if not np.isfinite(step) or step >= correlation - tolerance:
            break
        correlations -= step * direction
        correlation -= step

    return np.asarray(entered, dtype=np.int64)


def _segment_costs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered_y = y - y.mean()
    prefix_sum = np.concatenate(([0.0], np.cumsum(centered_y)))
    prefix_squares = np.concatenate(([0.0], np.cumsum(centered_y**2)))
    return prefix_sum, prefix_squares


def reduced_dynamic_programming(
    y: np.ndarray, candidate_points: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Minimise piecewise-constant squared error over candidate subsets."""
    n_samples = y.size
    candidates = np.unique(np.asarray(candidate_points, dtype=np.int64))
    candidates = candidates[(candidates > 0) & (candidates < n_samples)]
    boundaries = np.concatenate(([0], candidates, [n_samples]))
    n_boundaries = boundaries.size
    max_segments = n_boundaries - 1
    prefix_sum, prefix_squares = _segment_costs(y)

    costs = np.full((max_segments + 1, n_boundaries), np.inf)
    previous = np.full((max_segments + 1, n_boundaries), -1, dtype=np.int64)
    costs[0, 0] = 0.0

    # Cache each possible interval cost once, then vectorize predecessor
    # comparisons. The DP still explores every subset size in O(K^3) time.
    segment_costs = np.full((n_boundaries, n_boundaries), np.inf)
    for right in range(1, n_boundaries):
        starts = boundaries[:right]
        end = boundaries[right]
        sums = prefix_sum[end] - prefix_sum[starts]
        squares = prefix_squares[end] - prefix_squares[starts]
        segment_costs[:right, right] = np.maximum(
            squares - sums**2 / (end - starts), 0.0
        )

    for n_segments in range(1, max_segments + 1):
        for right in range(n_segments, n_boundaries):
            lefts = slice(n_segments - 1, right)
            values = costs[n_segments - 1, lefts] + segment_costs[lefts, right]
            left = n_segments - 1 + int(np.argmin(values))
            costs[n_segments, right] = values[left - n_segments + 1]
            previous[n_segments, right] = left

    objective_path = costs[1:, -1]
    change_points_by_count: list[np.ndarray] = []
    for n_segments in range(1, max_segments + 1):
        right = n_boundaries - 1
        change_points: list[int] = []
        for remaining_segments in range(n_segments, 0, -1):
            left = previous[remaining_segments, right]
            if left < 0:
                raise RuntimeError("Reduced dynamic programming found no path.")
            if remaining_segments > 1:
                change_points.append(int(boundaries[left]))
            right = left
        change_points_by_count.append(np.asarray(change_points[::-1], dtype=np.int64))

    return objective_path, tuple(change_points_by_count)


def select_change_count(objective_path: np.ndarray, nu: float) -> int:
    """Apply Cachalot's relative-improvement rule to the objective path."""
    for n_change_points in range(objective_path.size - 1):
        current = float(objective_path[n_change_points])
        following = float(objective_path[n_change_points + 1])
        if current == 0 or following / current >= 1.0 - nu:
            return n_change_points
    return objective_path.size - 1
