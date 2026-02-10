"""Experimental fuzzy F1 metric for change point detection.

This module introduces a differentiable alternative to the classic F1-score.
Instead of relying on a hard margin around each change point, it evaluates
predictions with a Gaussian reward that decays smoothly as the predicted
change point drifts away from the ground truth. The default configuration uses
the same Gaussian width for every change point, derived from a single fraction
of the series length so that no event is implicitly favoured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np

from .base import BaseMetric
from .change_point_detection import labels_to_change_points


def _ensure_list(values: Sequence[int] | np.ndarray | None) -> List[int]:
	"""Helper to coerce arrays and iterables to a sorted list of integers."""

	if values is None:
		return []
	if isinstance(values, np.ndarray):
		values = values.tolist()
	return sorted(int(v) for v in values)


def _strip_boundaries(cps: Sequence[int], series_length: int | None) -> List[int]:
	"""Remove boundary markers (0 and series length) from a list of change points."""

	if not cps:
		return []
	stripped = []
	for point in cps:
		if point == 0:
			continue
		if series_length is not None and point == series_length:
			continue
		stripped.append(point)
	return stripped


def _infer_series_length(cps: Sequence[int]) -> int | None:
	"""Best-effort inference of the total number of time steps.

	We assume that the last element corresponds to the series length when
	boundaries are included (as in the existing F1 metric). If this assumption
	fails, ``None`` is returned and downstream helpers fall back to defaults.
	"""

	if not cps:
		return None
	return int(max(cps))


def _adaptive_sigma(cps: Sequence[int], index: int, series_length: int | None, *, min_sigma: float = 1.0) -> float:
	"""Legacy adaptive sigma based on neighbour spacing."""

	current = cps[index]
	neighbours: List[float] = []

	if index > 0:
		neighbours.append(current - cps[index - 1])
	if index < len(cps) - 1:
		neighbours.append(cps[index + 1] - current)

	if neighbours:
		sigma = max(min_sigma, 0.5 * (sum(neighbours) / len(neighbours)))
	else:
		if series_length is None:
			sigma = max(min_sigma, 10.0)
		else:
			sigma = max(min_sigma, 0.1 * series_length)

	return float(sigma)


def _fixed_sigma(series_length: int | None, *, sigma_fraction: float, min_sigma: float = 1.0) -> float:
	"""Sigma shared by all change points, proportional to the series length."""

	if series_length is None:
		raise ValueError(
			"Cannot compute fixed Gaussian width without a defined series length."
		)

	return float(max(min_sigma, sigma_fraction * series_length))


def _local_sigma(
	cps: Sequence[int],
	index: int,
	series_length: int | None,
	*,
	adaptive: bool,
	sigma_fraction: float,
	min_sigma: float,
) -> float:
	"""Unified entry point used by internal helpers."""

	if adaptive:
		return _adaptive_sigma(cps, index, series_length, min_sigma=min_sigma)
	return _fixed_sigma(series_length, sigma_fraction=sigma_fraction, min_sigma=min_sigma)


def _gaussian_weight(distance: float, sigma: float) -> float:
	"""Compute the Gaussian reward for a given distance and spread."""

	return float(np.exp(-0.5 * (distance / sigma) ** 2))


@dataclass
class GaussianMatchResult:
	"""Container storing intermediate results of the fuzzy matching."""

	matched_weight: float
	used_true: List[int]
	used_pred: List[int]


def _max_weight_matching(
	true_cps: Sequence[int],
	pred_cps: Sequence[int],
	sigma_getter: Callable[[int, int], float],
) -> GaussianMatchResult:
	"""Greedy bipartite matching that maximises the summed Gaussian weight.

	A full Hungarian assignment would require an additional dependency. The
	greedy alternative is sufficient here because the Gaussian reward already
	penalises distant matches, making the cost surface well behaved. We build
	every possible (true, predicted) pair, sort them by weight in descending
	order, then keep the best non-conflicting matches.
	"""

	if not true_cps or not pred_cps:
		return GaussianMatchResult(0.0, [], [])

	pairs = []
	for i, true_cp in enumerate(true_cps):
		sigma = sigma_getter(i, true_cp)
		for j, pred_cp in enumerate(pred_cps):
			weight = _gaussian_weight(abs(pred_cp - true_cp), sigma)
			if weight > 0:
				pairs.append((weight, i, j))

	pairs.sort(reverse=True, key=lambda item: item[0])

	used_true: List[int] = []
	used_pred: List[int] = []
	matched_weight = 0.0

	for weight, i, j in pairs:
		if i in used_true or j in used_pred:
			continue
		matched_weight += weight
		used_true.append(i)
		used_pred.append(j)

	return GaussianMatchResult(matched_weight, used_true, used_pred)


class GaussianF1Score(BaseMetric):
	"""Gaussian-weighted alternative to the classic F1 score.

	The metric operates in three conceptual steps:

	1. **Preparation** – convert optional label sequences into change point
		lists, remove boundary markers, and infer the series length.
	2. **Gaussian matching** – every true change point is associated with a
		Gaussian of width ``sigma_fraction * n`` (clamped below by ``min_sigma``).
		Predictions are rewarded according to that shared kernel and a greedy
		assignment keeps the best non-overlapping pairs.
	3. **Soft precision & recall** – derive precision and recall from the sum of
		Gaussian rewards, yielding a fuzzy F1 in :math:`[0, 1]`.

	Special cases are handled explicitly:

	* **No ground-truth change point** – if the data really is stationary and no
		change points are predicted either, we return the perfect score 1.0.
		Conversely, predicting spurious changes yields a zero score.
	* **Single change point** – the Gaussian spread still follows the global
		fraction, ensuring a consistent reward scale across all events.
	"""

	def __init__(
		self,
		*,
		sigma_fraction: float = 0.01,
		min_sigma: float = 1.0,
		adaptive_sigma: bool = False,
		convert_labels_to_segments: bool = False,
	):
		super().__init__()
		self.sigma_fraction = float(sigma_fraction)
		self.min_sigma = float(min_sigma)
		self.adaptive_sigma = bool(adaptive_sigma)
		self.convert_labels_to_segments = bool(convert_labels_to_segments)

	def compute(
		self,
		y_true: Iterable[int] | np.ndarray,
		y_pred: Iterable[int] | np.ndarray,
	) -> dict[str, float]:
		"""Return the Gaussian-weighted precision, recall, and F1 score."""

		if self.convert_labels_to_segments:
			y_true = labels_to_change_points(y_true)
			y_pred = labels_to_change_points(y_pred)

		y_true_list = _ensure_list(y_true)
		y_pred_list = _ensure_list(y_pred)

		series_length = _infer_series_length(y_true_list)
		true_cps = _strip_boundaries(y_true_list, series_length)
		pred_cps = _strip_boundaries(y_pred_list, series_length)

		# Handle degenerate scenarios explicitly for clarity.
		if not true_cps and not pred_cps:
			return {
				"score": 1.0,
				"precision": 1.0,
				"recall": 1.0,
				"matched_weight": 0.0,
			}

		if not true_cps and pred_cps:
			return {
				"score": 0.0,
				"precision": 0.0,
				"recall": 0.0,
				"matched_weight": 0.0,
			}

		if true_cps and not pred_cps:
			return {
				"score": 0.0,
				"precision": 0.0,
				"recall": 0.0,
				"matched_weight": 0.0,
			}

		sigma_cache: dict[int, float] = {}

		def _sigma_for(index: int, cp: int) -> float:
			if self.adaptive_sigma:
				key = index
			else:
				key = -1
			if key not in sigma_cache:
				sigma_cache[key] = _local_sigma(
					true_cps,
					index,
					series_length,
					adaptive=self.adaptive_sigma,
					sigma_fraction=self.sigma_fraction,
					min_sigma=self.min_sigma,
				)
			return sigma_cache[key]

		match = _max_weight_matching(true_cps, pred_cps, _sigma_for)

		matched_weight = match.matched_weight
		n_pred = len(pred_cps)
		n_true = len(true_cps)

		precision = matched_weight / n_pred if n_pred else 0.0
		recall = matched_weight / n_true if n_true else 0.0

		if precision + recall == 0:
			f1 = 0.0
		else:
			f1 = 2 * precision * recall / (precision + recall)

		return {
			"score": float(f1),
			"precision": float(precision),
			"recall": float(recall),
			"matched_weight": float(matched_weight),
		}

