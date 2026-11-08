"""State detection metrics, comparing dense label sequences of equal length.

ARI, NMI and AMI are those of scikit-learn. WARI and WNMI weight each time
point by its distance to the nearest true boundary, and SMS matches predicted
to true states before classifying the errors (arXiv:2510.23261).
"""

from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from ._support import (
    _boundary_distances,
    _check_intervals,
    _covers_all,
    _island_boundary_distances,
    _mask,
)
from .base import BaseMetric

"""
WARI & WNMI scores
Code adapted from scikit-learn original implementation
https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/metrics/cluster/_supervised.py
Authors: The scikit-learn developers
SPDX-License-Identifier: BSD-3-Clause
"""


def _map_predicted_labels(labels_true, labels_pred, mask=None):
    """
    Maps predicted labels to match true labels using the Hungarian algorithm,
    ensuring the number of unique output labels equals the number of unique
    input predicted labels.

    Parameters:
        labels_true: numpy array of ground truth labels.
        labels_pred: numpy array of predicted labels.
        mask: optional boolean array of the scored points. The assignment is
            then fitted on these points only, and applied to every point.

    Returns:
        mapped_pred: numpy array of predicted labels after mapping.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    original_shape = labels_pred.shape
    if mask is not None:
        all_unique_pred = np.unique(labels_pred)
        final_map = _label_map(labels_true[mask], labels_pred[mask], all_unique_pred)
        mapped_pred_flat = np.array([final_map[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    if labels_pred.size == 0:
        return np.array([])

    all_unique_pred = np.unique(labels_pred)

    # Handle cases where true labels are empty or comparison is not possible
    if labels_true.size == 0:
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    # Determine common length for comparison
    compare_len = min(len(labels_pred.ravel()), len(labels_true.ravel()))
    if compare_len == 0:  # No overlap to compare
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    true_comp = labels_true.ravel()[:compare_len]
    pred_comp = labels_pred.ravel()[:compare_len]
    final_map = _label_map(true_comp, pred_comp, all_unique_pred)

    # Apply the final map to the original full predicted sequence
    mapped_pred_flat = np.array([final_map[val] for val in labels_pred.ravel()])
    mapped_pred = mapped_pred_flat.reshape(original_shape)

    return mapped_pred


def _label_map(true_comp, pred_comp, all_unique_pred):
    """Hungarian assignment of the predicted labels of ``pred_comp`` to the true
    labels of ``true_comp``, completed so that every label of
    ``all_unique_pred`` gets a distinct output label."""
    unique_pred_comp = np.unique(pred_comp)
    unique_true_comp = np.unique(true_comp)

    # Cost matrix: negative overlap
    cost_matrix = np.zeros((len(unique_pred_comp), len(unique_true_comp)))
    for i, p_label in enumerate(unique_pred_comp):
        for j, t_label in enumerate(unique_true_comp):
            overlap = np.sum((pred_comp == p_label) & (true_comp == t_label))
            cost_matrix[i, j] = -overlap

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    hungarian_map = {
        unique_pred_comp[i]: unique_true_comp[j]
        for i, j in zip(row_ind, col_ind, strict=True)
    }

    # Build the final mapping ensuring unique outputs for all unique predicted labels
    final_map = {}
    used_targets = set()

    # 1. Apply Hungarian assignments where the target is available
    for p_label, t_label in hungarian_map.items():
        if t_label not in used_targets:
            final_map[p_label] = t_label
            used_targets.add(t_label)
        # else: Target taken, p_label will be handled in step 2

    # 2. Assign remaining unique predicted labels
    next_new_label = 0
    for p_label in all_unique_pred:
        if p_label not in final_map:  # If not assigned yet
            # Try assigning p_label to itself if available
            if p_label not in used_targets:
                final_map[p_label] = p_label
                used_targets.add(p_label)
            # Otherwise, find the next available non-negative integer
            else:
                while next_new_label in used_targets:
                    next_new_label += 1
                final_map[p_label] = next_new_label
                used_targets.add(next_new_label)

    return final_map


def _error_type(error_label, true_atomicity, p0, p1, t0, t1):
    if true_atomicity == 1:
        # Delay
        # Check if the predicted error label matches either the preceding or succeeding true label.
        if (
            t0 is not None
            and p0 is not None
            and t0 == error_label
            and p0 == error_label
        ) or (
            t1 is not None
            and p1 is not None
            and t1 == error_label
            and p1 == error_label
        ):
            return "delay"

        # Isolation
        else:
            return "isolation"

    elif true_atomicity == 2:
        # Transition
        # True labels contains two different labels.
        return "transition"

    elif true_atomicity > 2:
        # Missing
        # True labels contain more than two different labels.
        return "missing"


def _normalized_block_boundary_distance(
    true_boundaries, block_start, block_end, n_samples
):
    """Compute normalized distance to the nearest real change point.

    Implements the distance term ``d`` of Algorithm 1: the minimum distance
    between the error block ``[block_start, block_end]`` and the surrounding
    real change points, normalized by the half the total sequence length ``n``.
    """
    if n_samples <= 0:
        return 0.0

    boundaries = np.asarray(true_boundaries, dtype=float)
    if boundaries.size == 0:
        return 0.0

    idx_prev = np.searchsorted(boundaries, block_start, side="right") - 1
    if idx_prev < 0:
        idx_prev = 0
    prev_boundary = boundaries[idx_prev]

    idx_next = np.searchsorted(boundaries, block_end, side="right")
    if idx_next >= boundaries.size:
        idx_next = boundaries.size - 1
    next_boundary = boundaries[idx_next]

    gap_prev = max(0.0, block_start - prev_boundary)
    gap_next = max(0.0, next_boundary - block_end)
    min_gap = min(gap_prev, gap_next)

    return 2.0 * min_gap / float(n_samples)


def _compute_boundaries_symmetrical(labels):
    """
    Computes the indices of the boundaries in the label sequence.

    Parameters:
        labels: numpy array of labels.

    Returns:
        List of indices where the label changes.
    """
    boundaries = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            # boundaries.append(i-1)
            boundaries.append(i)
    return boundaries


def _atomicity(sequence):
    # Remove consecutive duplicates while preserving order
    if len(sequence) > 0:
        reduced_segment = [sequence[0]]
        for x in sequence[1:]:
            if x != reduced_segment[-1]:
                reduced_segment.append(x)
        sequence_atomicity = len(reduced_segment)
    else:
        sequence_atomicity = 0
    return sequence_atomicity


def _state_matching_score(
    labels_true,
    labels_pred,
    weights,
    return_mapped=False,
    return_errors=False,
    islands=None,
):
    """
    Computes a new State Matching Score (SMS) based on identifying and classifying
    error segments between true and predicted label sequences after optimal mapping.

    ``islands`` lists the scored intervals ``[a, b)`` (default: the whole
    series). The mapping is fitted on their points, error segments stop at
    their edges, neighbours beyond an edge are absent, the edges anchor the
    boundary distance, and the losses are divided by the number of scored
    points.
    """
    n = len(labels_true)
    if n == 0:
        return None  # Return None for empty input
    if islands is None:
        islands = [(0, n)]
        mapped_pred = _map_predicted_labels(labels_true, labels_pred)
    else:
        mapped_pred = _map_predicted_labels(
            labels_true, labels_pred, mask=_mask(islands, n)
        )
    n_scored = sum(b - a for a, b in islands)

    total_penalty = (
        0.0  # Renamed from error_score for clarity, represents the sum of penalties
    )
    total_error_length = 0
    errors_list = [] if return_errors else None

    for island_start, island_end in islands:
        total_error_length, total_penalty = _island_errors(
            labels_true,
            mapped_pred,
            weights,
            island_start,
            island_end,
            n_scored,
            total_error_length,
            total_penalty,
            errors_list,
        )

    # Calculate the final score
    # The total error is the length of incorrect points plus the weighted penalty
    total_weighted_error = total_error_length + total_penalty
    score = 1.0 - total_weighted_error / n_scored

    # Return based on flags
    if return_mapped and return_errors:
        return score, mapped_pred, errors_list
    elif return_mapped:
        return score, mapped_pred
    elif return_errors:
        return score, errors_list
    else:
        return score


def _island_errors(
    labels_true,
    mapped_pred,
    weights,
    a,
    b,
    n_scored,
    total_error_length,
    total_penalty,
    errors_list,
):
    """Error segments of the scored interval ``[a, b)``, added to the totals."""
    true_boundaries = np.unique(
        [a] + [a + k for k in _compute_boundaries_symmetrical(labels_true[a:b])] + [b]
    )
    return_errors = errors_list is not None

    i = a
    while i < b:
        if labels_true[i] != mapped_pred[i]:
            # Start of a potential error segment
            start_index = i
            error_label = mapped_pred[i]
            j = i + 1
            # Extend the segment as long as the error persists with the same predicted label
            while (
                j < b
                and labels_true[j] != mapped_pred[j]
                and mapped_pred[j] == error_label
            ):
                j += 1
            end_index = j - 1

            # Extract the segment of true labels corresponding to the error segment
            true_labels_segment = np.asarray(labels_true[start_index : end_index + 1])
            true_atomicity = _atomicity(true_labels_segment)

            # Calculate segment size
            segment_size = end_index - start_index + 1
            total_error_length += segment_size

            # Determine neighbor values for predicted labels
            p0 = mapped_pred[start_index - 1] if start_index > a else None
            p1 = mapped_pred[end_index + 1] if end_index < b - 1 else None

            # Determine neighbor values for true labels
            t0 = labels_true[start_index - 1] if start_index > a else None
            t1 = labels_true[end_index + 1] if end_index < b - 1 else None

            # Call the error_type function with the required arguments
            err_type = _error_type(error_label, true_atomicity, p0, p1, t0, t1)

            segment_penalty = 0.0
            weight = weights.get(err_type, 1.0)  # Default weight if type not in dict

            if err_type == "delay":
                segment_penalty = weight * segment_size

            elif err_type == "transition" or err_type == "isolation":
                distance = _normalized_block_boundary_distance(
                    true_boundaries, start_index, end_index, n_scored
                )
                segment_penalty = segment_size * (weight * distance)

            elif err_type == "missing":
                segment_penalty = (
                    segment_size
                    * weight
                    * (1.0 + 3.0 * (weight - 1.0) / float(true_atomicity))
                )

            total_penalty += segment_penalty

            if return_errors:
                errors_list.append(
                    {
                        "type": err_type,
                        "start": start_index,
                        "end": end_index,
                        "size": segment_size,
                        "penalty": segment_penalty,  # Store the calculated penalty for this segment
                    }
                )

            # Move the main loop index past this segment
            i = j
        else:
            # No error at this index, move to the next
            i += 1

    return total_error_length, total_penalty


class StateMatchingScore(BaseMetric):
    """Computes the State Matching Score (SMS)."""

    DEFAULT_WEIGHTS = {
        "delay": 0.1,
        "transition": 0.3,
        "isolation": 0.8,
        "missing": 0.5,
    }

    def __init__(self, weights: dict[str, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.weights = (
            dict(weights) if weights is not None else self.DEFAULT_WEIGHTS.copy()
        )

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        scored_intervals=None,
        **kwargs,
    ) -> dict[str, Any]:
        """Compute the State Matching Score.

        Predicted states are first mapped to true states by the Hungarian
        algorithm. Each maximal run of mismatched points with the same
        predicted label is then an error segment, typed by the true labels it
        covers: inside one true segment, ``"delay"`` if it continues the
        neighbouring state (a boundary placed late or early), else
        ``"isolation"``; ``"transition"`` if it covers one true boundary;
        ``"missing"`` if it covers several. The score is ``1 - (L + P) / n``,
        L the total length of the error segments and P their penalties,
        weighted by type with ``self.weights`` (isolation and transition also
        by the distance to the nearest true boundary).

        With ``scored_intervals`` (masked SMS), the mapping is fitted on the
        scored points only and applied to every point; error segments stop at
        the edges of the intervals, where neighbours count as absent; the
        edges serve as anchors of the boundary distance without being true
        boundaries; and the losses are divided by the number of scored points.
        The score therefore does not depend on the length of the unscored
        gaps, and an interval covering the whole series gives the plain SMS.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_timepoints,)
            Ground-truth state label of every time point.
        y_pred : np.ndarray of shape (n_timepoints,)
            Predicted state label of every time point. Labels are compared up
            to a permutation: their values carry no meaning.
        scored_intervals : sequence of (int, int), optional
            Half-open intervals ``[a, b)`` of the time points to score, in
            original coordinates, for a reference with unscored points
            (missing truth, unknown or background labels); see
            :func:`tsseg.metrics.scored_islands`. Default: every point.
        **kwargs
            ``return_mapped`` (bool, default False): also return the predicted
            labels after mapping. ``return_errors`` (bool, default False): also
            return the error segments.

        Returns
        -------
        dict
            ``"score"`` (float, 1.0 for empty inputs), plus ``"mapped_pred"``
            (np.ndarray) if ``return_mapped`` and ``"errors"`` (list of dict
            with keys ``type``, ``start``, ``end``, ``size``, ``penalty``) if
            ``return_errors``. With ``scored_intervals``, also ``"n_scored"``
            and ``"n_total"``, the numbers of scored and of all time points.

        Raises
        ------
        ValueError
            If ``scored_intervals`` holds no scored point, or if ``y_true``
            and ``y_pred`` differ in length when it is given.
        """
        labels_true = y_true
        labels_pred = y_pred

        return_mapped = kwargs.get("return_mapped", False)
        return_errors = kwargs.get("return_errors", False)

        islands = None
        if scored_intervals is not None:
            labels_true = np.asarray(labels_true)
            labels_pred = np.asarray(labels_pred)
            _check_same_length(labels_true, labels_pred)
            islands = _check_intervals(scored_intervals, len(labels_true))

        result = _state_matching_score(
            labels_true,
            labels_pred,
            self.weights,
            return_mapped=return_mapped,
            return_errors=return_errors,
            islands=islands,
        )

        output: dict[str, Any] = {}
        if return_mapped and return_errors:
            score, mapped_pred, errors_list = result
            output["mapped_pred"] = mapped_pred
            output["errors"] = errors_list
        elif return_mapped:
            score, mapped_pred = result
            output["mapped_pred"] = mapped_pred
        elif return_errors:
            score, errors_list = result
            output["errors"] = errors_list
        else:
            score = result

        if score is None:
            score = 1.0
        else:
            score = float(score)

        output["score"] = score
        if islands is not None:
            output["n_scored"] = sum(b - a for a, b in islands)
            output["n_total"] = len(labels_true)
        return output


def _check_same_length(labels_true: np.ndarray, labels_pred: np.ndarray) -> None:
    if labels_true.ndim != 1 or labels_true.shape != labels_pred.shape:
        raise ValueError(
            "y_true and y_pred must be one-dimensional and of the same length, "
            f"got shapes {labels_true.shape} and {labels_pred.shape}"
        )


def _restrict(y_true, y_pred, scored_intervals):
    """The scored points of ``y_true`` and ``y_pred``, and their mask."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if scored_intervals is None:
        return y_true, y_pred, None
    _check_same_length(y_true, y_pred)
    islands = _check_intervals(scored_intervals, len(y_true))
    mask = _mask(islands, len(y_true))
    return y_true[mask], y_pred[mask], islands


class AdjustedRandIndex(BaseMetric):
    """Computes the Adjusted Rand Index (ARI)."""

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        scored_intervals=None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute the Adjusted Rand Index with :func:`sklearn.metrics.adjusted_rand_score`.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_timepoints,)
            Ground-truth state label of every time point.
        y_pred : np.ndarray of shape (n_timepoints,)
            Predicted state label of every time point. Labels are compared up
            to a permutation: their values carry no meaning.
        scored_intervals : sequence of (int, int), optional
            Half-open intervals ``[a, b)`` of the time points to score, in
            original coordinates, for a reference with unscored points
            (missing truth, unknown or background labels); see
            :func:`tsseg.metrics.scored_islands`. Default: every point. The score is
            that of the scored points.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            ``{"score": float}``.
        """
        labels_true, labels_pred, _ = _restrict(y_true, y_pred, scored_intervals)
        return {"score": adjusted_rand_score(labels_true, labels_pred)}


class NormalizedMutualInformation(BaseMetric):
    """Computes the Normalized Mutual Information (NMI)."""

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        scored_intervals=None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute the Normalized Mutual Information with :func:`sklearn.metrics.normalized_mutual_info_score`.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_timepoints,)
            Ground-truth state label of every time point.
        y_pred : np.ndarray of shape (n_timepoints,)
            Predicted state label of every time point. Labels are compared up
            to a permutation: their values carry no meaning.
        scored_intervals : sequence of (int, int), optional
            Half-open intervals ``[a, b)`` of the time points to score, in
            original coordinates, for a reference with unscored points
            (missing truth, unknown or background labels); see
            :func:`tsseg.metrics.scored_islands`. Default: every point. The score is
            that of the scored points.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            ``{"score": float}``.
        """
        labels_true, labels_pred, _ = _restrict(y_true, y_pred, scored_intervals)
        return {"score": normalized_mutual_info_score(labels_true, labels_pred)}


class AdjustedMutualInformation(BaseMetric):
    """Computes the Adjusted Mutual Information (AMI)."""

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        scored_intervals=None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute the Adjusted Mutual Information with :func:`sklearn.metrics.adjusted_mutual_info_score`.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_timepoints,)
            Ground-truth state label of every time point.
        y_pred : np.ndarray of shape (n_timepoints,)
            Predicted state label of every time point. Labels are compared up
            to a permutation: their values carry no meaning.
        scored_intervals : sequence of (int, int), optional
            Half-open intervals ``[a, b)`` of the time points to score, in
            original coordinates, for a reference with unscored points
            (missing truth, unknown or background labels); see
            :func:`tsseg.metrics.scored_islands`. Default: every point. The score is
            that of the scored points.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            ``{"score": float}``.
        """
        labels_true, labels_pred, _ = _restrict(y_true, y_pred, scored_intervals)
        return {"score": adjusted_mutual_info_score(labels_true, labels_pred)}


def _compute_boundary_distances(labels: np.ndarray) -> np.ndarray:
    """Distance of each point to the nearest label change or end of the series;
    ``n`` everywhere when the labels never change."""
    labels = np.asarray(labels)
    n = len(labels)
    if not np.any(np.diff(labels) != 0):
        return np.full(n, n)
    return _boundary_distances(labels, np.array([0, n]))


def _distance_weights(metric, y_true, y_pred, scored_intervals):
    """Weights of the WARI and WNMI, and the points they score.

    With ``scored_intervals``, the distances are measured in original
    coordinates, to the true changes inside each interval and to its edges,
    before the unscored points are left out.
    """
    if scored_intervals is None:
        return y_true, y_pred, metric.distance_func(_compute_boundary_distances(y_true))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_same_length(y_true, y_pred)
    islands = _check_intervals(scored_intervals, len(y_true))
    if _covers_all(islands, len(y_true)):
        return y_true, y_pred, metric.distance_func(_compute_boundary_distances(y_true))
    mask = _mask(islands, len(y_true))
    distances = _island_boundary_distances(y_true, islands)[mask]
    return y_true[mask], y_pred[mask], metric.distance_func(distances)


def _linear_distance(distances: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Linear distance transformation."""
    return 1 + alpha * distances


def weighted_contingency_matrix(
    labels_true, labels_pred, weights, *, eps=None, sparse=False, dtype=np.float64
):
    """Build a weighted contingency matrix."""
    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]

    contingency = sp.coo_matrix(
        (weights, (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=dtype,
    )

    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            contingency += eps

    return contingency


def weighted_pair_confusion_matrix(labels_true, labels_pred, weights):
    """Compute the weighted pair confusion matrix."""
    contingency = weighted_contingency_matrix(
        labels_true, labels_pred, weights, sparse=True
    )
    return _pair_confusion_from_contingency(contingency, np.sum(weights))


def _pair_confusion_from_contingency(contingency, total_weight):
    """Pair confusion matrix of a sparse weighted contingency table."""
    row_sum = np.ravel(contingency.sum(axis=1))
    col_sum = np.ravel(contingency.sum(axis=0))

    sum_squares = np.sum(contingency.data**2)

    C = np.empty((2, 2), dtype=np.float64)
    C[1, 1] = sum_squares - total_weight
    C[0, 1] = contingency.dot(col_sum).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(row_sum).sum() - sum_squares
    C[0, 0] = total_weight**2 - C[0, 1] - C[1, 0] - sum_squares

    return C


def _is_one_to_one(contingency):
    """Whether no row and no column of a sparse table has two non-zero cells."""
    table = contingency.tocoo()
    nonzero = table.data != 0
    per_row = np.bincount(table.row[nonzero], minlength=table.shape[0])
    per_col = np.bincount(table.col[nonzero], minlength=table.shape[1])
    return bool(np.all(per_row <= 1) and np.all(per_col <= 1))


def weighted_adjusted_rand_score(labels_true, labels_pred, weights):
    """Compute the Weighted Adjusted Rand Index (WARI).

    The ARI formula applied to weighted counts: each cell of the contingency
    table sums the weights of its time points, and the pair counts derived
    from it are real numbers, used as they are.

    Two labelings that agree up to a renaming of the states score exactly
    ``1.0``. This case is read from the contingency table (no row and no
    column with two non-zero cells), not from the pair counts, whose rounding
    errors grow with the length of the series.
    """
    contingency = weighted_contingency_matrix(
        labels_true, labels_pred, weights, sparse=True
    )
    if _is_one_to_one(contingency):
        return 1.0

    (tn, fp), (fn, tp) = _pair_confusion_from_contingency(contingency, np.sum(weights))

    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denominator == 0:
        return 1.0

    return float(2.0 * (tp * tn - fn * fp) / denominator)


class WeightedAdjustedRandIndex(BaseMetric):
    """Computes the Weighted Adjusted Rand Index (WARI)."""

    def __init__(self, distance_func: str = "linear", alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if distance_func == "linear":
            self.distance_func = lambda d: _linear_distance(d, alpha=self.alpha)
        else:
            self.distance_func = distance_func
        self.alpha = alpha

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        scored_intervals=None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute the Weighted Adjusted Rand Index.

        Each time point is weighted by ``self.distance_func`` of its distance
        to the nearest boundary of ``y_true`` (the series ends included); the
        default weight is ``1 + alpha * distance``.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_timepoints,)
            Ground-truth state label of every time point.
        y_pred : np.ndarray of shape (n_timepoints,)
            Predicted state label of every time point. Labels are compared up
            to a permutation: their values carry no meaning.
        scored_intervals : sequence of (int, int), optional
            Half-open intervals ``[a, b)`` of the time points to score, in
            original coordinates, for a reference with unscored points
            (missing truth, unknown or background labels); see
            :func:`tsseg.metrics.scored_islands`. Default: every point. The distances
            are then measured in original coordinates, to the true changes
            inside each interval and to its edges, before the other points are
            left out.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            ``{"score": float}``.
        """
        labels_true, labels_pred, weights = _distance_weights(
            self, y_true, y_pred, scored_intervals
        )

        wari_score = weighted_adjusted_rand_score(labels_true, labels_pred, weights)

        return {"score": wari_score}


class WeightedNormalizedMutualInformation(BaseMetric):
    """Computes the Weighted Normalized Mutual Information (WNMI)."""

    def __init__(
        self,
        distance_func: str = "linear",
        alpha: float = 0.1,
        average_method: str = "arithmetic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if distance_func == "linear":
            self.distance_func = lambda d: _linear_distance(d, alpha=self.alpha)
        else:
            self.distance_func = distance_func
        self.alpha = alpha
        self.average_method = average_method

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        scored_intervals=None,
        **kwargs,
    ) -> dict[str, float]:
        """Compute the Weighted Normalized Mutual Information.

        Each time point is weighted by ``self.distance_func`` of its distance
        to the nearest boundary of ``y_true`` (the series ends included); the
        default weight is ``1 + alpha * distance``.

        Parameters
        ----------
        y_true : np.ndarray of shape (n_timepoints,)
            Ground-truth state label of every time point.
        y_pred : np.ndarray of shape (n_timepoints,)
            Predicted state label of every time point. Labels are compared up
            to a permutation: their values carry no meaning.
        scored_intervals : sequence of (int, int), optional
            Half-open intervals ``[a, b)`` of the time points to score, in
            original coordinates, for a reference with unscored points
            (missing truth, unknown or background labels); see
            :func:`tsseg.metrics.scored_islands`. Default: every point. The distances
            are then measured in original coordinates, to the true changes
            inside each interval and to its edges, before the other points are
            left out.
        **kwargs
            Ignored.

        Returns
        -------
        dict
            ``{"score": float}``.
        """
        labels_true, labels_pred, weights = _distance_weights(
            self, y_true, y_pred, scored_intervals
        )

        wnmi_score = weighted_normalized_mutual_info_score(
            labels_true, labels_pred, weights, average_method=self.average_method
        )

        return {"score": wnmi_score}


def weighted_mutual_info_score(labels_true, labels_pred, weights):
    """Compute the Weighted Mutual Information (WMI)."""
    contingency = weighted_contingency_matrix(
        labels_true, labels_pred, weights, sparse=True
    )

    # Calculate the MI for the two clusterings
    nzx, nzy, nz_val = sp.find(contingency)
    contingency_sum = contingency.sum()

    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    log_contingency_nm = np.log(nz_val)
    log_outer = np.log(pi[nzx]) + np.log(pj[nzy])

    mi = nz_val * (log_contingency_nm - np.log(contingency_sum)) - nz_val * (
        log_outer - 2 * np.log(contingency_sum)
    )

    return mi.sum() / contingency_sum


def weighted_entropy(labels, weights):
    """Compute the weighted entropy of a labeling."""
    _, labels_idx = np.unique(labels, return_inverse=True)
    total_weight = np.sum(weights)

    if total_weight == 0:
        return 0.0

    label_weights = np.bincount(labels_idx, weights=weights)

    # Filter out zero weights to avoid log(0)
    nz_label_weights = label_weights[label_weights > 0]

    if nz_label_weights.size == 0:
        return 0.0

    prob = nz_label_weights / total_weight
    return -np.sum(prob * np.log(prob))


def weighted_normalized_mutual_info_score(
    labels_true, labels_pred, weights, *, average_method="arithmetic"
):
    """Compute the Weighted Normalized Mutual Information (WNMI)."""
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    if (
        classes.shape[0] == clusters.shape[0] == 1
        or classes.shape[0] == clusters.shape[0] == 0
    ):
        return 1.0

    mi = weighted_mutual_info_score(labels_true, labels_pred, weights)

    if mi == 0:
        return 0.0

    h_true = weighted_entropy(labels_true, weights)
    h_pred = weighted_entropy(labels_pred, weights)

    if average_method == "min":
        normalizer = min(h_true, h_pred)
    elif average_method == "geometric":
        normalizer = np.sqrt(h_true * h_pred)
    elif average_method == "arithmetic":
        normalizer = (h_true + h_pred) / 2
    elif average_method == "max":
        normalizer = max(h_true, h_pred)
    else:
        raise ValueError(
            "average_method should be one of 'min', 'geometric', 'arithmetic', 'max'"
        )

    if normalizer == 0.0:
        return 0.0

    return mi / normalizer
