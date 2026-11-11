"""Cachalot change point detector."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import Closed, Interval, ParamDef
from ._core import (
    lasso_candidate_points,
    reduced_dynamic_programming,
    select_change_count,
)

__all__ = ["CachalotDetector"]


class CachalotDetector(BaseSegmenter):
    """Catch level changes with Lasso candidates and reduced dynamic programming.

    Cachalot writes a piecewise-constant signal as an intercept plus step
    functions at every possible change point. The L1 penalty is applied to
    the step coefficients. Cachalot (Harchaoui and Lévy-Leduc) takes the first
    ``max_candidates`` variables on the Lasso path, finds the best
    piecewise-constant fit over subsets of those candidates by dynamic
    programming, and selects the number of changes from the relative
    improvement of the residual sum of squares.

    The published method is univariate. For multivariate input, each channel
    is processed independently and the returned change points are the union
    of the per-channel estimates. This channel-wise extension is not a
    shared-change-point multivariate model such as group LARS.

    Parameters
    ----------
    max_candidates : int, default=50
        Maximum number of Lasso-path change-point candidates per channel.
        The actual limit is capped at ``n_timepoints - 1``.
    nu : float, default=0.01
        Relative-improvement threshold in the model-selection rule. The
        selected count is the first ``k`` for which
        ``J(k + 1) / J(k) >= 1 - nu``.
    axis : int, default=0
        Axis of ``X`` that represents time.

    Attributes
    ----------
    change_points_ : np.ndarray
        Sorted union of the selected change points across channels.
    candidate_points_ : tuple of np.ndarray
        Lasso-path candidates for each channel.
    objective_paths_ : tuple of np.ndarray
        Minimum residual sums of squares for zero through the maximum
        candidate-supported number of change points, per channel.

    Notes
    -----
    The step-function columns are centered to leave the signal level
    unpenalized, matching the total-variation formulation in the paper. Their
    Gram matrix is computed from the tail lengths, so the Lasso path does not
    materialize a dense ``n_timepoints x n_timepoints`` design matrix. The
    reduced dynamic program costs ``O(K^3)`` after selecting ``K`` candidates;
    the active-set path costs ``O(n K^2)`` in its usual regime.

    The 2007 paper's toy example prints the Lasso candidate set and objective
    table, but not the sampled noise vector or a random seed. Its published
    Monte Carlo tables likewise do not give seeds, so those individual
    numerical realizations cannot be regenerated exactly.

    References
    ----------
    .. [1] Harchaoui, Z. and Lévy-Leduc, C., "Catching Change-points with
       Lasso", Advances in Neural Information Processing Systems 20, 2007.
    .. [2] Harchaoui, Z. and Lévy-Leduc, C., "Multiple Change-Point
       Estimation With a Total Variation Penalty", Journal of the American
       Statistical Association 105(492):1480-1493, 2010.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    _parameter_schema = {
        "max_candidates": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Maximum Lasso-path change-point candidates per channel.",
        ),
        "nu": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Relative-improvement threshold in the selection rule.",
        ),
    }

    def __init__(self, *, max_candidates: int = 50, nu: float = 0.01, axis: int = 0):
        self.max_candidates = max_candidates
        self.nu = nu
        super().__init__(axis=axis)

    def _predict(self, X):
        if X.shape[0] < 2:
            empty = np.empty(0, dtype=np.int64)
            self.candidate_points_ = tuple(empty.copy() for _ in range(X.shape[1]))
            self.objective_paths_ = tuple(
                np.zeros(1, dtype=float) for _ in range(X.shape[1])
            )
            self.change_points_ = empty
            return empty

        candidates_by_channel = []
        objective_paths = []
        selected_by_channel = []
        for channel in X.T:
            candidates = lasso_candidate_points(channel, self.max_candidates)
            objective_path, change_points = reduced_dynamic_programming(
                channel, candidates
            )
            n_change_points = select_change_count(objective_path, self.nu)
            candidates_by_channel.append(candidates)
            objective_paths.append(objective_path)
            selected_by_channel.append(change_points[n_change_points])

        self.candidate_points_ = tuple(candidates_by_channel)
        self.objective_paths_ = tuple(objective_paths)
        self.change_points_ = np.unique(np.concatenate(selected_by_channel)).astype(
            np.int64
        )
        return self.change_points_
