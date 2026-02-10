from __future__ import annotations

import numpy as np
from ..base import BaseSegmenter
from ..utils import multivariate_l2_norm, aggregate_change_points
from scipy.special import gammaln, logsumexp

from tsseg.algorithms.bocd.bayesian_models import offline_changepoint_detection


class BOCDDetector(BaseSegmenter):
    """Bayesian change-point detector based on offline inference.

    This implementation wraps the offline Bayesian change-point detection
    heuristic introduced by Fearnhead (2006). It integrates out the mean and
    variance of each segment under a conjugate Normal-Gamma prior and selects
    change points by thresholding the posterior probability of a boundary.

    Parameters
    ----------
    hazard_lambda : float, default=300
        Expected run length. Internally converted to a constant hazard
        probability of ``1 / hazard_lambda``.
    mu : float, default=0.0
        Prior mean of the segment observations.
    kappa : float, default=1.0
        Strength of the prior mean (normal precision).
    alpha : float, default=1.0
        Prior shape of the inverse-gamma distribution over the variance.
    beta : float, default=1.0
        Prior scale of the inverse-gamma distribution over the variance.
    truncate : int, default=-40
        Log probability truncation used by the dynamic programme. Values
        further than ``10**truncate`` below the running total are skipped.
    cp_prob_threshold : float, default=0.05
        Minimum posterior probability required to accept a change point.
    min_distance : int, default=25
        Minimum distance (in samples) enforced between successive change
        points.
    max_cps : int | None, default=None
        Optional cap on the number of change points to return. ``None`` keeps
        all candidates above the probability threshold.
    multivariate_strategy : str, default="l2"
        Strategy for handling multivariate data:

        - ``"l2"`` – reduce to univariate via L2 norm across channels.
        - ``"ensembling"`` – run BOCD independently on each channel and
          aggregate the detected change points.
    tolerance : int | float, default=0
        Tolerance for aggregating change points in the ensembling strategy.
        If a float in ``(0, 1)`` it is interpreted as a fraction of the
        signal length.
    axis : int, default=0
        Time axis in the input array.
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

    def __init__(
        self,
        hazard_lambda: float = 300,
        mu: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        truncate: int = -40,
        cp_prob_threshold: float = 0.05,
        min_distance: int = 25,
        max_cps: int | None = None,
        multivariate_strategy: str = "l2",
        tolerance: int | float = 0,
        axis: int = 0,
    ):
        self.hazard_lambda = float(hazard_lambda)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.truncate = int(truncate)
        self.cp_prob_threshold = float(cp_prob_threshold)
        self.min_distance = int(max(min_distance, 1))
        self.max_cps = max_cps if max_cps is None else int(max_cps)
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        super().__init__(axis=axis)

    def _fit(self, X, y=None):  # noqa: D401 - no training necessary
        return self

    def _predict(self, X):
        # After BaseSeriesEstimator._preprocess_series, X is always a
        # np.ndarray with time on axis 0 (shape: (n_samples,) or
        # (n_samples, n_channels)).
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        n_samples, n_channels = data.shape

        if n_samples < 2:
            return np.empty(0, dtype=np.int64)

        if n_channels > 1 and self.multivariate_strategy == "ensembling":
            return self._predict_ensembling(data)

        # Univariate or L2 reduction
        if n_channels > 1:
            signal = multivariate_l2_norm(data)
        else:
            signal = data.ravel()

        return self._run_bocd(signal, n_samples)

    def _predict_ensembling(self, data: np.ndarray) -> np.ndarray:
        """Run BOCD independently on each channel and aggregate results."""
        n_samples, n_channels = data.shape
        all_cps: list[int] = []

        for d in range(n_channels):
            cps = self._run_bocd(data[:, d], n_samples)
            all_cps.extend(cps.tolist())

        if not all_cps:
            return np.empty(0, dtype=np.int64)

        n_cp = self.max_cps if self.max_cps is not None else len(all_cps)
        return aggregate_change_points(
            all_cps,
            n_cp=n_cp,
            tolerance=self.tolerance,
            signal_len=n_samples,
        )

    def _run_bocd(self, signal: np.ndarray, n_samples: int) -> np.ndarray:
        """Run offline BOCD on a univariate signal and return change points."""
        hazard_prob = 1.0 / max(self.hazard_lambda, 1.0)
        hazard_prob = float(np.clip(hazard_prob, 1e-6, 1.0 - 1e-6))

        def _prior(run_length: int) -> float:
            _ = run_length  # unused, but kept for API compatibility
            return np.log(hazard_prob)

        log_likelihood = _StudentTLogMarginal(
            mu0=self.mu, kappa0=self.kappa, alpha0=self.alpha, beta0=self.beta
        )

        _, _, log_pcp = offline_changepoint_detection(
            signal, _prior, log_likelihood, truncate=self.truncate
        )

        if log_pcp.size == 0:
            return np.empty(0, dtype=np.int64)

        log_cp_probs = logsumexp(log_pcp, axis=0)
        finite_mask = np.isfinite(log_cp_probs)
        if not finite_mask.any():
            return np.empty(0, dtype=np.int64)

        normaliser = logsumexp(log_cp_probs[finite_mask])
        cp_probabilities = np.zeros(log_cp_probs.shape[0], dtype=float)
        cp_probabilities[finite_mask] = np.exp(
            log_cp_probs[finite_mask] - normaliser
        )

        return _select_change_points(
            cp_probabilities,
            threshold=self.cp_prob_threshold,
            min_distance=self.min_distance,
            max_cps=self.max_cps,
            n_samples=n_samples,
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        return {
            "hazard_lambda": 100,
            "cp_prob_threshold": 0.02,
            "min_distance": 20,
        }


class _StudentTLogMarginal:
    """Normal-Gamma marginal likelihood for a univariate Gaussian segment.

    Results are cached per ``(t, s)`` pair to avoid redundant computation
    during the :math:`O(n^2)` dynamic programme.
    """

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        if kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0:
            raise ValueError("kappa, alpha and beta must be strictly positive")
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self._cache: dict[tuple[int, int], float] = {}
        self._cached_data_id: int | None = None

    def pdf(self, data: np.ndarray, t: int, s: int) -> float:
        # Invalidate cache when the underlying data array changes.
        data_id = id(data)
        if data_id != self._cached_data_id:
            self._cache.clear()
            self._cached_data_id = data_id

        key = (t, s)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._compute(data, t, s)
        self._cache[key] = result
        return result

    def _compute(self, data: np.ndarray, t: int, s: int) -> float:
        if s <= t:
            return -np.inf

        segment = data[t:s]
        n = segment.size
        if n == 0:
            return -np.inf

        sample_mean = segment.mean()
        centered = segment - sample_mean
        sample_ss = np.dot(centered, centered)

        kappa_n = self.kappa0 + n
        alpha_n = self.alpha0 + 0.5 * n
        beta_n = (
            self.beta0
            + 0.5 * sample_ss
            + (self.kappa0 * n * (sample_mean - self.mu0) ** 2) / (2 * kappa_n)
        )

        return (
            gammaln(alpha_n)
            - gammaln(self.alpha0)
            + self.alpha0 * np.log(self.beta0)
            - alpha_n * np.log(beta_n)
            + 0.5 * (np.log(self.kappa0) - np.log(kappa_n))
            - 0.5 * n * np.log(np.pi)
        )


def _select_change_points(
    probabilities: np.ndarray,
    threshold: float,
    min_distance: int,
    max_cps: int | None,
    n_samples: int,
) -> np.ndarray:
    if probabilities.size == 0:
        return np.empty(0, dtype=np.int64)

    order = np.argsort(probabilities)[::-1]
    selected: list[int] = []
    for idx in order:
        prob = probabilities[idx]
        if prob < threshold:
            break

        cp_position = idx + 1  # boundary is after sample idx
        if cp_position <= 0 or cp_position >= n_samples:
            continue

        if selected and np.min(np.abs(np.asarray(selected) - cp_position)) < min_distance:
            continue

        selected.append(cp_position)
        if max_cps is not None and len(selected) >= max_cps:
            break

    selected.sort()
    return np.asarray(selected, dtype=np.int64)