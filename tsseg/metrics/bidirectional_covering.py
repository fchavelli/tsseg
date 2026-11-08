"""Bidirectional Covering metric for change-point segmentation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ._support import (
    _check_intervals,
    _covers_all,
    _directional_covering,
    _masked_covering,
    _n_supported,
)
from .base import BaseMetric
from .change_point_detection import (
    _ensure_boundaries,
    _prepare_change_points,
    _require_length_for_support,
)


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
        *,
        n_timepoints: int | None = None,
        scored_intervals: Sequence[tuple[int, int]] | None = None,
    ) -> dict[str, float]:
        """Compute the bidirectional covering of two segmentations.

        Parameters
        ----------
        y_true, y_pred:
            True and predicted change points (or label sequences with
            ``convert_labels_to_segments=True``). Without ``n_timepoints``,
            the series length is the largest change point of both lists, so
            ``y_true`` must end with it.
        n_timepoints:
            Length of the series. When given, the change points may include
            or omit the boundaries ``0`` and ``n_timepoints``. When omitted, a
            ``UserWarning`` is emitted (unless ``convert_labels_to_segments``
            is set).
        scored_intervals:
            Half-open intervals ``[a, b)`` of the time points where the
            reference can be judged (masked BiC), for a reference with gaps of
            unknown truth; see :func:`tsseg.metrics.scored_islands`. True and
            predicted segments are clipped to each interval of at least two
            points, matched inside it only, and the two directional coverings
            are weighted by the scored duration. Change points outside the
            intervals count for nothing. Requires ``n_timepoints`` or
            ``convert_labels_to_segments``. Default: the whole series.

        Returns
        -------
        dict
            ``score``, ``ground_truth_covering`` and ``prediction_covering``;
            with ``scored_intervals``, also ``n_scored`` (scored duration),
            ``n_supported`` (change-point positions that can be judged) and
            ``n_total``.

        Raises
        ------
        ValueError
            If ``scored_intervals`` has no interval of at least two points.
        """
        if scored_intervals is not None:
            _require_length_for_support(
                n_timepoints, self.convert_labels_to_segments, type(self).__name__
            )
        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()
        else:
            y_true = list(y_true)

        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()
        else:
            y_pred = list(y_pred)

        y_true, y_pred = _prepare_change_points(
            y_true,
            y_pred,
            n_timepoints=n_timepoints,
            convert_labels_to_segments=self.convert_labels_to_segments,
            metric_name=type(self).__name__,
        )

        if scored_intervals is not None:
            n = y_true[-1]
            islands = _check_intervals(scored_intervals, n)
            if not _covers_all(islands, n):
                ground_truth_covering, prediction_covering, duration = _masked_covering(
                    y_true, y_pred, islands
                )
                return {
                    "score": self._aggregation(
                        ground_truth_covering, prediction_covering
                    ),
                    "ground_truth_covering": ground_truth_covering,
                    "prediction_covering": prediction_covering,
                    "n_scored": duration,
                    "n_supported": _n_supported(islands),
                    "n_total": n,
                }

        y_true, y_pred = _ensure_boundaries(y_true, y_pred)

        gt_segments = self._compute_segments(y_true)
        pred_segments = self._compute_segments(y_pred)

        ground_truth_covering = self._directional_covering(gt_segments, pred_segments)
        prediction_covering = self._directional_covering(pred_segments, gt_segments)

        score = self._aggregation(ground_truth_covering, prediction_covering)

        output = {
            "score": score,
            "ground_truth_covering": ground_truth_covering,
            "prediction_covering": prediction_covering,
        }
        if scored_intervals is not None:
            output.update(n_scored=n, n_supported=n - 1, n_total=n)
        return output

    @staticmethod
    def _compute_segments(change_points: Sequence[int]) -> list[tuple[int, int]]:
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
        source_segments: Iterable[tuple[int, int]],
        target_segments: Iterable[tuple[int, int]],
    ) -> float:
        return _directional_covering(source_segments, target_segments)


__all__ = ["BidirectionalCovering"]
