"""ChangeFinder: two-stage outlier-to-change-point detection.

Implements the ChangeFinder algorithm from:
    Takeuchi & Yamanishi, "A Unifying Framework for Detecting Outliers and
    Change Points from Time Series", IEEE TKDE, 2006.

The algorithm operates in two stages:
    1. An SDAR model learns the time series online.  Each observation receives
       an outlier score (log-loss or quadratic loss).  A moving average of
       these scores produces a smoothed score series.
    2. A second SDAR model learns the smoothed score series.  A second moving
       average yields the final change-point score at each time step.

Change points are then selected by peak-picking on the final score curve.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import argrelextrema

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    Interval,
    ParamDef,
    StrOptions,
)
from ..utils import aggregate_change_points, multivariate_l2_norm
from .sdar import SDAR


class ChangeFinderDetector(BaseSegmenter):
    """ChangeFinder change-point detector (Takeuchi & Yamanishi, 2006).

    Two-stage online method that reduces change-point detection to outlier
    detection.  An AR model with exponential discounting (SDAR) computes
    outlier scores; a moving average smooths them; a second SDAR + moving
    average produces a change-point score curve.  Peaks in the curve are
    returned as change points.

    Parameters
    ----------
    order : int, default=5
        AR order for both SDAR stages.
    discount : float, default=0.005
        Discounting rate *r* in (0, 1).  Controls how quickly the model
        forgets past observations.
    smooth_window : int, default=7
        Length *T* of the moving-average windows applied after each SDAR
        stage.
    score : str, default="logarithmic"
        Scoring function: ``"logarithmic"`` (negative log-likelihood) or
        ``"quadratic"`` (squared prediction error).
    n_cps : int or None, default=None
        Number of change points to return.  If ``None``, all peaks above
        ``threshold`` are returned.
    threshold : float or None, default=None
        Minimum score for a peak to be accepted as a change point.  When
        ``None``, a data-driven threshold of ``mean + 2*std`` of the score
        curve is used.
    min_distance : int, default=10
        Minimum number of samples between successive change points.
    multivariate_strategy : str, default="l2"
        Strategy for multivariate inputs: ``"l2"`` reduces to univariate
        via L2 norm; ``"ensembling"`` runs independently per channel and
        aggregates.
    tolerance : int or float, default=0
        Tolerance for aggregating change points across channels (ensembling
        strategy only).
    axis : int, default=0
        Time axis.

    References
    ----------
    .. [1] J. Takeuchi and K. Yamanishi, "A unifying framework for detecting
       outliers and change points from time series," IEEE TKDE, vol. 18,
       no. 4, pp. 482-492, 2006.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "order": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="AR order for SDAR models.",
        ),
        "discount": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Discounting rate r in (0, 1).",
        ),
        "smooth_window": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Moving-average window length T.",
        ),
        "score": ParamDef(
            constraint=StrOptions({"logarithmic", "quadratic"}),
            description="Scoring function for outlier scores.",
        ),
        "n_cps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of change points to return.",
            nullable=True,
        ),
        "threshold": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Minimum peak score to accept as change point.",
            nullable=True,
        ),
        "min_distance": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum samples between successive change points.",
        ),
        "multivariate_strategy": ParamDef(
            constraint=StrOptions({"l2", "ensembling"}),
            description="Strategy for multivariate data.",
        ),
        "tolerance": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Tolerance for aggregating CPs in ensembling.",
        ),
        "_cross_constraints": [
            DataDependent(
                "order < n_samples",
                "AR order must be smaller than the series length.",
            ),
            DataDependent(
                "smooth_window < n_samples",
                "Smooth window must be smaller than the series length.",
            ),
        ],
    }

    def __init__(
        self,
        order: int = 5,
        discount: float = 0.005,
        smooth_window: int = 7,
        score: str = "logarithmic",
        n_cps: int | None = None,
        threshold: float | None = None,
        min_distance: int = 10,
        multivariate_strategy: str = "l2",
        tolerance: int | float = 0,
        axis: int = 0,
    ) -> None:
        self.order = order
        self.discount = discount
        self.smooth_window = smooth_window
        self.score = score
        self.n_cps = n_cps
        self.threshold = threshold
        self.min_distance = min_distance
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # Core change-point scoring
    # ------------------------------------------------------------------

    def _changefinder_scores(self, signal: np.ndarray) -> np.ndarray:
        """Run the two-stage ChangeFinder pipeline on a 1-D signal.

        Returns the final change-point score curve (same length as *signal*).
        """
        n = len(signal)
        k = self.order
        T = self.smooth_window
        use_log = self.score == "logarithmic"

        # ---- Stage 1: SDAR on raw signal ----
        sdar1 = SDAR(order=k, discount=self.discount)
        init_len = min(max(2 * k, 30), n)
        sdar1._init_from_batch(signal[:init_len])

        scores1 = np.zeros(n)
        for t in range(init_len, n):
            x_hat, sigma = sdar1.update(signal[t])
            if use_log:
                scores1[t] = sdar1.log_loss(signal[t], x_hat, sigma)
            else:
                scores1[t] = sdar1.quadratic_loss(signal[t], x_hat)

        # Back-fill the init region with the median score so that stage 2
        # does not see an artificial jump from 0 to real scores.
        valid = scores1[init_len:]
        if len(valid) > 0:
            fill_val = float(np.median(valid))
            scores1[:init_len] = fill_val

        # ---- Moving average (stage 1 → smoothed scores) ----
        y = self._moving_average(scores1, T)

        # ---- Stage 2: SDAR on smoothed score series ----
        # Start stage 2 after a warm-up that covers init + smoothing
        warmup = init_len + T
        sdar2 = SDAR(order=k, discount=self.discount)
        init_len2 = min(max(2 * k, 30), n - warmup) if warmup < n else 0
        if init_len2 < k + 1:
            # Not enough data for a meaningful second stage
            return scores1

        sdar2._init_from_batch(y[warmup : warmup + init_len2])
        start2 = warmup + init_len2

        scores2 = np.zeros(n)
        for t in range(start2, n):
            y_hat, sigma2 = sdar2.update(y[t])
            if use_log:
                scores2[t] = sdar2.log_loss(y[t], y_hat, sigma2)
            else:
                scores2[t] = sdar2.quadratic_loss(y[t], y_hat)

        # ---- Moving average (stage 2 → final CP scores) ----
        cp_scores = self._moving_average(scores2, T)
        return cp_scores

    @staticmethod
    def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """Causal moving average with a rectangular window."""
        if window <= 1:
            return x.copy()
        kernel = np.ones(window) / window
        # Use 'full' mode then trim to preserve alignment
        smoothed = np.convolve(x, kernel, mode="full")[: len(x)]
        return smoothed

    # ------------------------------------------------------------------
    # Peak-based change-point selection
    # ------------------------------------------------------------------

    def _select_peaks(self, scores: np.ndarray) -> np.ndarray:
        """Select change points as peaks of the score curve."""
        n = len(scores)
        min_dist = max(self.min_distance, 1)

        # Find local maxima
        order = max(min_dist // 2, 1)
        peaks = argrelextrema(scores, np.greater, order=order)[0]

        if len(peaks) == 0:
            return np.empty(0, dtype=np.int64)

        # Filter by threshold
        if self.threshold is not None:
            thr = self.threshold
        else:
            # Data-driven: mean + 2*std of non-zero scores
            valid = scores[scores > 0]
            if len(valid) > 0:
                thr = float(np.mean(valid) + 2.0 * np.std(valid))
            else:
                thr = 0.0
        peaks = peaks[scores[peaks] > thr]

        if len(peaks) == 0:
            return np.empty(0, dtype=np.int64)

        # Sort by score (descending), enforce min_distance
        idx_sorted = peaks[np.argsort(scores[peaks])[::-1]]
        selected: list[int] = []
        for cp in idx_sorted:
            if cp < min_dist or cp > n - min_dist:
                continue
            if selected and min(abs(cp - s) for s in selected) < min_dist:
                continue
            selected.append(int(cp))
            if self.n_cps is not None and len(selected) >= self.n_cps:
                break

        # If n_cps requested but threshold too strict, relax
        if self.n_cps is not None and len(selected) < self.n_cps:
            remaining = [
                int(p)
                for p in idx_sorted
                if int(p) not in selected
                and min_dist <= p <= n - min_dist
                and (not selected or min(abs(p - s) for s in selected) >= min_dist)
            ]
            for cp in remaining:
                if selected and min(abs(cp - s) for s in selected) < min_dist:
                    continue
                selected.append(cp)
                if len(selected) >= self.n_cps:
                    break

        selected.sort()
        return np.asarray(selected, dtype=np.int64)

    # ------------------------------------------------------------------
    # BaseSegmenter interface
    # ------------------------------------------------------------------

    def _predict(self, X: np.ndarray) -> np.ndarray:
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        n_samples, n_channels = data.shape

        if n_samples < self.order + 2:
            return np.empty(0, dtype=np.int64)

        if n_channels > 1 and self.multivariate_strategy == "ensembling":
            return self._predict_ensembling(data)

        # Univariate or L2 reduction
        if n_channels > 1:
            signal = multivariate_l2_norm(data)
        else:
            signal = data.ravel()

        scores = self._changefinder_scores(signal)
        return self._select_peaks(scores)

    def _predict_ensembling(self, data: np.ndarray) -> np.ndarray:
        n_samples, n_channels = data.shape
        all_cps: list[int] = []

        for d in range(n_channels):
            scores = self._changefinder_scores(data[:, d])
            cps = self._select_peaks(scores)
            all_cps.extend(cps.tolist())

        if not all_cps:
            return np.empty(0, dtype=np.int64)

        n_cp = self.n_cps if self.n_cps is not None else len(all_cps)
        return aggregate_change_points(
            all_cps,
            n_cp=n_cp,
            tolerance=self.tolerance,
            signal_len=data.shape[0],
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        return {
            "order": 3,
            "discount": 0.05,
            "smooth_window": 5,
            "min_distance": 10,
        }
