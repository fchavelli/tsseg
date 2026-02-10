"""Bidirectional Covering metric for change-point segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .base import BaseMetric
from .change_point_detection import labels_to_change_points, _ensure_boundaries


@dataclass(frozen=True)
class _AggregationStrategy:
	name: str

	def __call__(self, coverage_a: float, coverage_b: float) -> float:
		coverage_a = float(coverage_a)
		coverage_b = float(coverage_b)

		if self.name == "harmonic":
			if coverage_a <= 0.0 or coverage_b <= 0.0:
				return 0.0
			return (2.0 * coverage_a * coverage_b) / (coverage_a + coverage_b)
		if self.name == "geometric":
			if coverage_a <= 0.0 or coverage_b <= 0.0:
				return 0.0
			return math.sqrt(coverage_a * coverage_b)
		if self.name == "arithmetic":
			return 0.5 * (coverage_a + coverage_b)
		if self.name == "min":
			return min(coverage_a, coverage_b)

		raise ValueError(f"Unknown aggregation strategy: {self.name}")


class BidirectionalCovering(BaseMetric):
	"""Bidirectional extension of the classical Covering metric.

	The classical Covering score only evaluates how well predicted segments cover
	the ground-truth segmentation. However, this directionality means that long
	predicted segments that cover the truth sparsely may still obtain a high
	score, even when the prediction introduces substantial over-segmentation.

	The bidirectional variant evaluates coverage in both directions:

	* ``ground_truth_covering`` mirrors the traditional definition where each
	  ground-truth interval is weighted by its duration and matched to the best
	  overlapping predicted interval via Intersection over Union (IoU).
	* ``prediction_covering`` swaps the roles. Each predicted segment is
	  weighted by its duration and matched to the best ground-truth overlap.

	The two directional scores are then aggregated using an F1-style harmonic
	mean by default. Alternative aggregation strategies (``geometric``,
	``arithmetic`` or ``min``) can be selected via the ``aggregation``
	argument. The resulting metric rewards segmentations that both cover the
	truth and avoid excessive over-segmentation.

	Parameters
	----------
	convert_labels_to_segments:
		When ``True``, the inputs are interpreted as label sequences and will
		be converted to change-points via
		:func:`tsseg.metrics.change_point_detection.labels_to_change_points`.
	aggregation:
		Name of the aggregation strategy used to combine the two directional
		covering scores. Supported values are ``"harmonic"`` (default),
		``"geometric"``, ``"arithmetic"`` and ``"min"``.
	kwargs:
		Forwarded to :class:`tsseg.metrics.base.BaseMetric`.
	"""

	def __init__(
		self,
		*,
		convert_labels_to_segments: bool = False,
		aggregation: str = "harmonic",
		**kwargs,
	) -> None:
		super().__init__(**kwargs)
		self.convert_labels_to_segments = convert_labels_to_segments
		aggregation = aggregation.lower().strip()
		valid = {"harmonic", "geometric", "arithmetic", "min"}
		if aggregation not in valid:
			raise ValueError(
				"aggregation must be one of {'harmonic', 'geometric', 'arithmetic', 'min'}"
			)
		self._aggregation = _AggregationStrategy(aggregation)

	def compute(
		self,
		y_true: Sequence[int] | np.ndarray,
		y_pred: Sequence[int] | np.ndarray,
	) -> Dict[str, float]:
		if isinstance(y_true, np.ndarray):
			y_true = y_true.tolist()
		else:
			y_true = list(y_true)

		if isinstance(y_pred, np.ndarray):
			y_pred = y_pred.tolist()
		else:
			y_pred = list(y_pred)

		if self.convert_labels_to_segments:
			y_true = labels_to_change_points(y_true)
			y_pred = labels_to_change_points(y_pred)

		y_true, y_pred = _ensure_boundaries(y_true, y_pred)

		gt_segments = self._compute_segments(y_true)
		pred_segments = self._compute_segments(y_pred)

		ground_truth_covering = self._directional_covering(gt_segments, pred_segments)
		prediction_covering = self._directional_covering(pred_segments, gt_segments)

		score = self._aggregation(ground_truth_covering, prediction_covering)

		return {
			"score": score,
			"ground_truth_covering": ground_truth_covering,
			"prediction_covering": prediction_covering,
		}

	@staticmethod
	def _compute_segments(change_points: Sequence[int]) -> List[Tuple[int, int]]:
		if not change_points:
			return []

		ordered = sorted(int(cp) for cp in change_points)
		return [
			(ordered[idx], ordered[idx + 1])
			for idx in range(len(ordered) - 1)
			if ordered[idx + 1] > ordered[idx]
		]

	@staticmethod
	def _directional_covering(
		source_segments: Iterable[Tuple[int, int]],
		target_segments: Iterable[Tuple[int, int]],
	) -> float:
		source_segments = list(source_segments)
		target_segments = list(target_segments)

		if not source_segments:
			return 1.0 if not target_segments else 0.0

		total_length = sum(max(0, end - start) for start, end in source_segments)
		if total_length <= 0:
			return 0.0

		if not target_segments:
			return 0.0

		# Ensure target segments are sorted for the sweep.
		target_segments.sort(key=lambda seg: seg[0])

		coverage_sum = 0.0
		target_len = len(target_segments)
		pointer = 0

		for src_start, src_end in source_segments:
			if src_end <= src_start:
				continue

			max_iou = 0.0

			while pointer < target_len and target_segments[pointer][1] <= src_start:
				pointer += 1

			idx = pointer
			while idx < target_len and target_segments[idx][0] < src_end:
				tgt_start, tgt_end = target_segments[idx]
				intersection = max(0, min(src_end, tgt_end) - max(src_start, tgt_start))
				if intersection > 0:
					union = (src_end - src_start) + (tgt_end - tgt_start) - intersection
					if union > 0:
						max_iou = max(max_iou, intersection / union)
				idx += 1

			coverage_sum += (src_end - src_start) * max_iou

		return coverage_sum / total_length


__all__ = ["BidirectionalCovering"]
