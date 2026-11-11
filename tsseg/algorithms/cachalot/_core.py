"""Numerical routines for the Cachalot change point method."""

from __future__ import annotations

import numpy as np

__all__ = [
    "lasso_candidate_points",
    "reduced_dynamic_programming",
    "select_change_count",
]


def _gram_columns(left: np.ndarray, right: np.ndarray, n_samples: int) -> np.ndarray:
    """Gram products of centered step columns with the given tail lengths."""
    return np.minimum(left[:, None], right[None, :]) - (
        left[:, None] * right[None, :] / n_samples
    )


def lasso_candidate_points(y: np.ndarray, max_candidates: int) -> np.ndarray:
    """Return change-point candidates entering the Lasso path, in entry order.

    The design columns are centered step functions, so the signal's level is
    unpenalized and only its jumps enter the L1 penalty. Their Gram products
    have a closed form, avoiding the dense ``n_samples x n_samples`` design
    matrix.
    """
    n_samples = y.size
    n_features = n_samples - 1
    limit = min(max_candidates, n_features)
    if limit <= 0:
        return np.empty(0, dtype=np.int64)

    centered_y = y - y.mean()
    tails = np.arange(n_features, 0, -1, dtype=float)
    correlations = -np.cumsum(centered_y)[:-1]
    coefficients = np.zeros(n_features)
    active: list[int] = []
    signs: list[float] = []
    entered: list[int] = []

    while len(entered) < limit:
        if not active:
            next_feature = int(np.argmax(np.abs(correlations)))
            if correlations[next_feature] == 0:
                break
            active.append(next_feature)
            signs.append(float(np.sign(correlations[next_feature])))
            entered.append(next_feature)
            continue

        active_array = np.asarray(active, dtype=np.int64)
        active_signs = np.asarray(signs)
        active_correlations = correlations[active_array]
        current_correlation = float(np.max(np.abs(active_correlations)))
        active_gram = _gram_columns(tails[active_array], tails[active_array], n_samples)
        try:
            least_squares_direction = np.linalg.solve(active_gram, active_signs)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("The Lasso active-set system is singular.") from exc

        direction_scale = float(active_signs @ least_squares_direction)
        if direction_scale <= 0 or not np.isfinite(direction_scale):
            raise RuntimeError("The Lasso active-set direction is not positive.")
        equiangular_scale = 1.0 / np.sqrt(direction_scale)
        active_direction = equiangular_scale * least_squares_direction
        all_active_gram = _gram_columns(tails, tails[active_array], n_samples)
        correlation_direction = all_active_gram @ active_direction

        inactive = np.ones(n_features, dtype=bool)
        inactive[active_array] = False
        inactive_indices = np.flatnonzero(inactive)
        entry_steps: list[tuple[float, int, float]] = []
        denominator_tolerance = (
            100 * np.finfo(float).eps * max(1.0, abs(equiangular_scale))
        )

        for numerator, denominator, sign in (
            (
                current_correlation - correlations[inactive_indices],
                equiangular_scale - correlation_direction[inactive_indices],
                1.0,
            ),
            (
                current_correlation + correlations[inactive_indices],
                equiangular_scale + correlation_direction[inactive_indices],
                -1.0,
            ),
        ):
            valid = denominator > denominator_tolerance
            steps = np.full(inactive_indices.size, np.inf)
            steps[valid] = numerator[valid] / denominator[valid]
            valid &= steps >= -denominator_tolerance
            entry_steps.extend(
                (max(0.0, float(steps[i])), int(inactive_indices[i]), sign)
                for i in np.flatnonzero(valid)
            )

        entry_gamma = min((step[0] for step in entry_steps), default=np.inf)
        coefficient_scale = max(1.0, float(np.max(np.abs(coefficients))))
        coefficient_tolerance = 100 * np.finfo(float).eps * coefficient_scale
        drop_steps: list[tuple[float, int]] = []
        for active_position, (coefficient, direction) in enumerate(
            zip(coefficients[active_array], active_direction, strict=True)
        ):
            if abs(coefficient) > coefficient_tolerance and coefficient * direction < 0:
                drop_steps.append((float(-coefficient / direction), active_position))

        drop_gamma = min((step[0] for step in drop_steps), default=np.inf)
        end_gamma = current_correlation / equiangular_scale
        gamma = min(entry_gamma, drop_gamma, end_gamma)
        if not np.isfinite(gamma) or gamma < 0:
            break

        coefficients[active_array] += gamma * active_direction
        correlations -= gamma * correlation_direction
        step_tolerance = max(1e-12, abs(gamma) * 1e-10)
        reached_endpoint = abs(end_gamma - gamma) <= step_tolerance

        if abs(drop_gamma - gamma) <= step_tolerance:
            dropped = [
                position
                for step, position in drop_steps
                if abs(step - gamma) <= step_tolerance
            ]
            for position in reversed(dropped):
                coefficients[active[position]] = 0.0
                active.pop(position)
                signs.pop(position)

        if abs(entry_gamma - gamma) <= step_tolerance and not reached_endpoint:
            entering, sign = min(
                (
                    (feature, sign)
                    for step, feature, sign in entry_steps
                    if abs(step - entry_gamma) <= step_tolerance
                ),
                key=lambda item: item[0],
            )
            active.append(entering)
            signs.append(sign)
            if entering not in entered:
                entered.append(entering)
            continue

        if reached_endpoint:
            break

    return np.asarray(entered, dtype=np.int64) + 1


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

    for n_segments in range(1, max_segments + 1):
        for right in range(n_segments, n_boundaries):
            for left in range(n_segments - 1, right):
                start, end = boundaries[left], boundaries[right]
                segment_sum = prefix_sum[end] - prefix_sum[start]
                segment_squares = prefix_squares[end] - prefix_squares[start]
                segment_cost = segment_squares - segment_sum**2 / (end - start)
                value = costs[n_segments - 1, left] + max(float(segment_cost), 0.0)
                if value < costs[n_segments, right]:
                    costs[n_segments, right] = value
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
