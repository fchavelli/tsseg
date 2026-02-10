import warnings
import numpy as np
from typing import List, Dict, Union

from .base import BaseMetric

'''
F1 & Covering scores
Code adapted from TSSB (time-series-segmentation-benchmark)
https://github.com/ermshaua/time-series-segmentation-benchmark/blob/95a62b8e1e4e380313f187544c38f3400c1773e5/tssb/evaluation.py
Authors: Van den Burg, G.J.J. and Williams, C.K.I. from The Alan Turing Institute
'''

def labels_to_change_points(labels):
    """
    Convert label sequence into change points (CPs).
    
    Parameters:
        labels (list or np.array): Label sequence.
    
    Returns:
        list: Change points (CPs) including start (0) and end (n).
    """
    n = len(labels)
    # np.where finds indices where the condition is true.
    # np.diff(labels) is non-zero at each change point.
    cp_indices = np.where(np.diff(labels) != 0)[0] + 1
    # Return a list that always includes the start (0) and end (n) points.
    return [0] + cp_indices.tolist() + [n]


def _ensure_boundaries(
    y_true: List[int], y_pred: List[int]
) -> tuple[List[int], List[int]]:
    """Ensure both change-point lists include ``0`` and the series length ``T``.

    The series length is inferred as ``max(max(y_true), max(y_pred))``.
    If either list is empty the other is used to determine ``T``.

    This is critical for the Covering metrics:  without boundaries the
    first and last segments are silently dropped, producing wrong scores.
    """
    if not y_true and not y_pred:
        return [0], [0]

    all_points = (y_true or []) + (y_pred or [])
    T = max(all_points)

    def _add_bounds(cps: List[int], T: int) -> List[int]:
        s = set(cps)
        s.add(0)
        s.add(T)
        return sorted(s)

    return _add_bounds(y_true, T), _add_bounds(y_pred, T)


def _strip_boundaries(change_points: List[int], series_length: int | None) -> List[int]:
    """Remove leading/trailing boundaries from a change-point sequence."""

    if not change_points:
        return []

    filtered = []
    for point in change_points:
        if point == 0:
            continue
        if series_length is not None and point == series_length:
            continue
        filtered.append(point)
    return filtered

def _true_positives(y_true: List[int], y_pred: List[int], margin: int) -> int:
    """
    Compute true positives without double counting.

    Args:
        y_true: List of true change point locations.
        y_pred: List of predicted change point locations.
        margin: Maximum allowed distance between a true and a predicted change-point.

    Returns:
        The number of true positives.
    """
    # Make a copy so we don't modify the original
    temp_y_pred = set(y_pred)
    tp = 0
    # Sort true change points to handle cases where multiple true CPs
    # are close to the same predicted CP.
    for true_cp in sorted(y_true):
        # Find all predictions that are within the margin
        close_preds = [p for p in temp_y_pred if abs(true_cp - p) <= margin]
        
        if not close_preds:
            continue
        
        # Find the closest prediction
        closest_pred = min(close_preds, key=lambda p: abs(true_cp - p))
        
        # We have a match
        tp += 1
        # Remove the matched prediction to avoid double counting
        temp_y_pred.remove(closest_pred)
        
    return tp

class F1Score(BaseMetric):
    """Computes the F1-score for change point detection."""

    def __init__(
        self,
        margin: Union[int, float] = 0.01,
        convert_labels_to_segments: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.convert_labels_to_segments = convert_labels_to_segments

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes the F1-score, precision, and recall.

        Args:
            y_true: List of true change points. The last element should be the
                    total number of time steps.
            y_pred: List of predicted change points.
        Returns:
            A dictionary with F1-score, precision, and recall.
        """
        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        if self.convert_labels_to_segments:
            y_true = labels_to_change_points(y_true)
            y_pred = labels_to_change_points(y_pred)

        if not y_true and not y_pred:
            return {
                "score": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            }

        if not y_true or not y_pred:
            return {
                "score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        series_length = y_true[-1] if y_true else None

        margin = self.margin
        if isinstance(margin, float):
            if not (0.0 <= margin <= 1.0):
                raise ValueError("Float margin must be between 0.0 and 1.0")
            if series_length is None:
                raise ValueError("Cannot derive absolute margin without a series length")
            computed_margin = int(margin * series_length)
            if margin > 0 and computed_margin == 0:
                warnings.warn(
                    "Computed margin equals 0 samples for the given series length; "
                    "the tolerance will have no effect.",
                    UserWarning,
                )
            margin = computed_margin

        margin = int(margin)
        if margin < 0:
            raise ValueError("Margin must be non-negative")

        filtered_true = _strip_boundaries(y_true, series_length)
        filtered_pred = _strip_boundaries(y_pred, series_length)

        if not filtered_true and not filtered_pred:
            return {
                "score": 1.0,
                "precision": 1.0,
                "recall": 1.0,
            }

        if not filtered_true or not filtered_pred:
            return {
                "score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        tp = _true_positives(filtered_true, filtered_pred, margin)

        precision = tp / len(filtered_pred)
        recall = tp / len(filtered_true)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "score": f1,
            "precision": precision,
            "recall": recall,
        }

class Covering(BaseMetric):
    """
    Computes the Covering score for a segmentation.

    The Covering metric evaluates how well the predicted segments cover the
    ground truth segments. It is calculated as a weighted sum of the maximum
    Intersection over Union (IoU) for each ground truth segment, where the
    weight is the length of the ground truth segment. This implementation is
    based on the logic proposed in various segmentation evaluation studies.
    """

    def __init__(self, convert_labels_to_segments: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.convert_labels_to_segments = convert_labels_to_segments

    def _compute_segments(self, cp_indices: List[int]) -> List[tuple[int, int]]:
        """
        Convert change points to segment intervals.

        Args:
            cp_indices: List of change point indices.

        """
        return [(cp_indices[i], cp_indices[i + 1]) for i in range(len(cp_indices) - 1)]

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes the Covering score.

        Args:
            y_true: Array of true change points. The last element should be the
                    total number of time steps.
            y_pred: Array of predicted change points.

        Returns:
            A dictionary with the Covering score.
        """
        if isinstance(y_true, np.ndarray):
            y_true = y_true.tolist()
        if isinstance(y_pred, np.ndarray):
            y_pred = y_pred.tolist()

        if self.convert_labels_to_segments:
            y_true = labels_to_change_points(y_true)
            y_pred = labels_to_change_points(y_pred)

        y_true, y_pred = _ensure_boundaries(y_true, y_pred)

        gt_segments = self._compute_segments(y_true)
        pred_segments = self._compute_segments(y_pred)

        covering_score = 0.0
        p_idx = 0
        p_len = len(pred_segments)

        for gt_start, gt_end in gt_segments:
            segment_size = gt_end - gt_start
            if segment_size == 0:
                continue

            max_iou = 0.0

            # Advance pointer in predicted segments to find potential overlaps
            while p_idx < p_len and pred_segments[p_idx][1] <= gt_start:
                p_idx += 1

            # Check all predicted segments that could overlap with the current ground truth segment
            temp_idx = p_idx
            while temp_idx < p_len and pred_segments[temp_idx][0] < gt_end:
                pred_start, pred_end = pred_segments[temp_idx]

                # Calculate intersection and union
                intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
                if intersection > 0:
                    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
                    iou = intersection / union if union > 0 else 0.0
                    max_iou = max(max_iou, iou)

                temp_idx += 1
            # Weight the maximum IoU by the segment size
            covering_score += segment_size * max_iou
        # Normalize by the total length of the ground truth segments
        total_length = sum(
            gt_end - gt_start for gt_start, gt_end in gt_segments if gt_end > gt_start
        )

        covering = covering_score / total_length if total_length > 0 else 0.0

        return {"score": covering}

class HausdorffDistance(BaseMetric):
    """Computes the Hausdorff distance between two sets of change points."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes the Hausdorff distance.

        Args:
            y_true: Array of true change points.
            y_pred: Array of predicted change points.

        Returns:
            A dictionary with the Hausdorff distance.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.size == 0 or y_pred.size == 0:
            return {"score": np.inf, "hausdorff_distance": np.inf}
            
        dist_matrix = np.abs(y_true[:, np.newaxis] - y_pred)
        
        h1 = np.max(np.min(dist_matrix, axis=1))
        h2 = np.max(np.min(dist_matrix, axis=0))

        distance = max(h1, h2)
        return {"score": distance, "hausdorff_distance": distance}
