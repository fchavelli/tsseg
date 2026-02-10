import numpy as np
from typing import Dict, Any
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp

from .base import BaseMetric

'''
WARI & WNMI scores
Code adapted from scikit-learn original implementation
https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73a86f5f11781a0e21f24c8f47979ec67/sklearn/metrics/cluster/_supervised.py
Authors: The scikit-learn developers
SPDX-License-Identifier: BSD-3-Clause
'''

def _map_predicted_labels(labels_true, labels_pred):
    """
    Maps predicted labels to match true labels using the Hungarian algorithm,
    ensuring the number of unique output labels equals the number of unique
    input predicted labels.

    Parameters:
        labels_true: numpy array of ground truth labels.
        labels_pred: numpy array of predicted labels.

    Returns:
        mapped_pred: numpy array of predicted labels after mapping.
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    original_shape = labels_pred.shape

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
    if compare_len == 0: # No overlap to compare
        mapping = {label: i for i, label in enumerate(all_unique_pred)}
        mapped_pred_flat = np.array([mapping[val] for val in labels_pred.ravel()])
        return mapped_pred_flat.reshape(original_shape)

    true_comp = labels_true.ravel()[:compare_len]
    pred_comp = labels_pred.ravel()[:compare_len]

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
    hungarian_map = {unique_pred_comp[i]: unique_true_comp[j] for i, j in zip(row_ind, col_ind)}

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
        if p_label not in final_map: # If not assigned yet
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

    # Apply the final map to the original full predicted sequence
    mapped_pred_flat = np.array([final_map[val] for val in labels_pred.ravel()])
    mapped_pred = mapped_pred_flat.reshape(original_shape)

    return mapped_pred

def _error_type(error_label, true_atomicity, p0, p1, t0, t1):
    if true_atomicity == 1:
        # Delay
        # Check if the predicted error label matches either the preceding or succeeding true label.
        if (t0 is not None and p0 is not None and t0 == error_label and p0 == error_label) or (t1 is not None and p1 is not None and t1 == error_label and p1 == error_label):
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

def _normalized_block_boundary_distance(true_boundaries, block_start, block_end, n_samples):
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
        if labels[i] != labels[i-1]:
            #boundaries.append(i-1)
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

def _state_matching_score(labels_true, labels_pred, weights, return_mapped=False, return_errors=False):
    """
    Computes a new State Matching Score (SMS) based on identifying and classifying
    error segments between true and predicted label sequences after optimal mapping.
    """
    n = len(labels_true)
    if n == 0:
        return None # Return None for empty input

    # Map predicted labels to true labels
    mapped_pred = _map_predicted_labels(labels_true, labels_pred)
    true_boundaries = np.unique([0] + _compute_boundaries_symmetrical(labels_true) + [n])

    i = 0
    total_penalty = 0.0 # Renamed from error_score for clarity, represents the sum of penalties
    total_error_length = 0
    errors_list = [] if return_errors else None

    while i < n:
        if labels_true[i] != mapped_pred[i]:
            # Start of a potential error segment
            start_index = i
            error_label = mapped_pred[i]
            j = i + 1
            # Extend the segment as long as the error persists with the same predicted label
            while j < n and labels_true[j] != mapped_pred[j] and mapped_pred[j] == error_label:
                j += 1
            end_index = j - 1

            # Extract the segment of true labels corresponding to the error segment
            true_labels_segment = np.asarray(labels_true[start_index : end_index + 1])
            true_atomicity = _atomicity(true_labels_segment)

            # Calculate segment size
            segment_size = end_index - start_index + 1
            total_error_length += segment_size

            # Determine neighbor values for predicted labels
            p0 = mapped_pred[start_index - 1] if start_index > 0 else None
            p1 = mapped_pred[end_index + 1] if end_index < n - 1 else None

            # Determine neighbor values for true labels
            t0 = labels_true[start_index - 1] if start_index > 0 else None
            t1 = labels_true[end_index + 1] if end_index < n - 1 else None

            # Call the error_type function with the required arguments
            err_type = _error_type(error_label, true_atomicity, p0, p1, t0, t1)

            segment_penalty = 0.0
            weight = weights.get(err_type, 1.0) # Default weight if type not in dict

            if err_type == "delay":
                segment_penalty = weight * segment_size

            elif err_type == "transition" or err_type == "isolation":
                distance = _normalized_block_boundary_distance(
                    true_boundaries, start_index, end_index, n
                )
                segment_penalty = segment_size * (weight * distance)

            elif err_type == "missing":
                segment_penalty = segment_size * weight * (1.0 + 3.0 * (weight - 1.0) / float(true_atomicity))

            total_penalty += segment_penalty

            if return_errors:
                errors_list.append({
                    'type': err_type,
                    'start': start_index,
                    'end': end_index,
                    'size': segment_size,
                    'penalty': segment_penalty # Store the calculated penalty for this segment
                })

            # Move the main loop index past this segment
            i = j
        else:
            # No error at this index, move to the next
            i += 1

    # Calculate the final score
    # The total error is the length of incorrect points plus the weighted penalty
    total_weighted_error = total_error_length + total_penalty
    score = 1.0 - total_weighted_error / n

    # Return based on flags
    if return_mapped and return_errors:
        return score, mapped_pred, errors_list
    elif return_mapped:
        return score, mapped_pred
    elif return_errors:
        return score, errors_list
    else:
        return score

class StateMatchingScore(BaseMetric):
    """Computes the State Matching Score (SMS)."""

    DEFAULT_WEIGHTS = {"delay": 0.1, "transition": 0.3, "isolation": 0.8, "missing": 0.5}

    def __init__(self, weights: Dict[str, float] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.weights = dict(weights) if weights is not None else self.DEFAULT_WEIGHTS.copy()

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        labels_true = y_true
        labels_pred = y_pred

        return_mapped = kwargs.get("return_mapped", False)
        return_errors = kwargs.get("return_errors", False)

        result = _state_matching_score(
            labels_true,
            labels_pred,
            self.weights,
            return_mapped=return_mapped,
            return_errors=return_errors,
        )

        output: Dict[str, Any] = {}
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
        return output

class AdjustedRandIndex(BaseMetric):
    """Computes the Adjusted Rand Index (ARI)."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        labels_true = y_true
        labels_pred = y_pred
        return {"score": adjusted_rand_score(labels_true, labels_pred)}

class NormalizedMutualInformation(BaseMetric):
    """Computes the Normalized Mutual Information (NMI)."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        labels_true = y_true
        labels_pred = y_pred
        return {"score": normalized_mutual_info_score(labels_true, labels_pred)}

class AdjustedMutualInformation(BaseMetric):
    """Computes the Adjusted Mutual Information (AMI)."""

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        labels_true = y_true
        labels_pred = y_pred
        return {"score": adjusted_mutual_info_score(labels_true, labels_pred)}

def _compute_boundary_distances(labels: np.ndarray) -> np.ndarray:
    """Computes distance to nearest boundary."""
    n = len(labels)
    boundaries = np.where(np.diff(labels) != 0)[0] + 1
    if len(boundaries) == 0:
        return np.full(n, n)
    
    boundaries = np.concatenate(([0], boundaries, [n]))
    
    distances = np.empty(n)
    for i in range(n):
        distances[i] = np.min(np.abs(i - boundaries))
        
    return distances

def _linear_distance(distances: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Linear distance transformation."""
    return 1 + alpha * distances

def weighted_contingency_matrix(labels_true, labels_pred, weights, *, eps=None, sparse=False, dtype=np.float64):
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
    contingency = weighted_contingency_matrix(labels_true, labels_pred, weights, sparse=True)
    
    total_weight = np.sum(weights)
    row_sum = np.ravel(contingency.sum(axis=1))
    col_sum = np.ravel(contingency.sum(axis=0))
    
    sum_squares = np.sum(contingency.data**2)
    
    C = np.empty((2, 2), dtype=np.float64)
    C[1, 1] = sum_squares - total_weight
    C[0, 1] = contingency.dot(col_sum).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(row_sum).sum() - sum_squares
    C[0, 0] = total_weight**2 - C[0, 1] - C[1, 0] - sum_squares
    
    return C

def weighted_adjusted_rand_score(labels_true, labels_pred, weights):
    """Compute the Weighted Adjusted Rand Index (WARI)."""
    (tn, fp), (fn, tp) = weighted_pair_confusion_matrix(labels_true, labels_pred, weights)
    
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    if fn == 0 and fp == 0:
        return 1.0

    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if denominator == 0:
        return 1.0
        
    return 2.0 * (tp * tn - fn * fp) / denominator

class WeightedAdjustedRandIndex(BaseMetric):
    """Computes the Weighted Adjusted Rand Index (WARI)."""

    def __init__(self, distance_func: str = "linear", alpha: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if distance_func == "linear":
            self.distance_func = lambda d: _linear_distance(d, alpha=self.alpha)
        else:
            self.distance_func = distance_func
        self.alpha = alpha

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        labels_true = y_true
        labels_pred = y_pred
        
        d_true = _compute_boundary_distances(labels_true)
        weights = self.distance_func(d_true)
        
        wari_score = weighted_adjusted_rand_score(labels_true, labels_pred, weights)
        
        return {"score": wari_score}

class WeightedNormalizedMutualInformation(BaseMetric):
    """Computes the Weighted Normalized Mutual Information (WNMI)."""

    def __init__(self, distance_func: str = "linear", alpha: float = 0.1, average_method: str = "arithmetic", **kwargs):
        super().__init__(**kwargs)
        if distance_func == "linear":
            self.distance_func = lambda d: _linear_distance(d, alpha=self.alpha)
        else:
            self.distance_func = distance_func
        self.alpha = alpha
        self.average_method = average_method

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        labels_true = y_true
        labels_pred = y_pred
        
        d_true = _compute_boundary_distances(labels_true)
        weights = self.distance_func(d_true)
        
        wnmi_score = weighted_normalized_mutual_info_score(labels_true, labels_pred, weights, average_method=self.average_method)
        
        return {"score": wnmi_score}

def weighted_mutual_info_score(labels_true, labels_pred, weights):
    """Compute the Weighted Mutual Information (WMI)."""
    contingency = weighted_contingency_matrix(labels_true, labels_pred, weights, sparse=True)
    
    # Calculate the MI for the two clusterings
    nzx, nzy, nz_val = sp.find(contingency)
    contingency_sum = contingency.sum()
    
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))
    
    log_contingency_nm = np.log(nz_val)
    log_outer = np.log(pi[nzx]) + np.log(pj[nzy])
    
    mi = (
        nz_val * (log_contingency_nm - np.log(contingency_sum))
        - nz_val * (log_outer - 2 * np.log(contingency_sum))
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

def weighted_normalized_mutual_info_score(labels_true, labels_pred, weights, *, average_method="arithmetic"):
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
