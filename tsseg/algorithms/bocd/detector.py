from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.special import gammaln

from tsseg.algorithms.bocd._offline import (
    _boundary_marginals,
    _GaussianSegments,
    _map_partition,
)

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)
from ..utils import aggregate_change_points, multivariate_l2_norm


class BOCDDetector(BaseSegmenter):
    """Bayesian change-point detector based on offline inference.

    Segment parameters are integrated out under a conjugate prior. Native
    multivariate inference includes the full covariance matrix. The default
    returns the joint maximum a posteriori partition; thresholded marginal
    boundary probabilities remain available.

    Parameters
    ----------
    hazard_lambda : float, default=1000
        Expected run length before conditioning on the candidate grid and
        minimum segment length. The unrestricted sample-time hazard is
        ``1 / hazard_lambda``.
    mu : float, default=0.0
        Prior mean of the segment observations.
    kappa : float, default=1.0
        Strength of the prior mean (normal precision).
    alpha : float, default=1.0
        Prior shape of the inverse-gamma distribution over the variance.
    beta : float, default=10.0
        Prior scale of the inverse-gamma distribution over the variance.
    truncate : int | None, default=None
        Optional legacy natural-log cutoff in marginal decoding. ``None``
        sums every term on the candidate grid. A finite cutoff can discard
        important terms and does not accelerate likelihood evaluation.
        Not used by MAP decoding.
    cp_prob_threshold : float, default=0.5
        Minimum probability of a peak in marginal decoding. Not used by MAP.
    min_distance : int, default=25
        Minimum segment length, including the first and last segments.
    max_cps : int | None, default=None
        Optional upper bound on the number of changes. MAP jointly optimises
        over partitions with at most this many boundaries. Marginal decoding
        caps the selected peaks. ``0`` returns none.
    multivariate_strategy : str, default="native"
        Strategy for handling multivariate data:

        - ``"native"`` – integrate the mean vector and full covariance.
        - ``"l2"`` – reduce to univariate via L2 norm across channels.
        - ``"ensembling"`` – run BOCD independently on each channel and
          aggregate the detected change points.
    tolerance : int | float, default=0
        Tolerance for aggregating change points in the ensembling strategy.
        If a float in ``(0, 1)`` it is interpreted as a fraction of the
        signal length.
    axis : int, default=0
        Time axis in the input array.
    decoding : {"map", "marginal"}, default="map"
        Joint MAP partition or thresholded marginal peaks.
    max_candidates : int | None, default=2000
        Maximum interior boundary candidates on a regular grid. All
        observations contribute to the likelihood. ``None`` permits every
        interior position. Inference is conditional on the candidate grid.
    standardize : bool, default=True
        Center and scale each channel on the whole series. Prior parameters
        are then in standardised units. Disable to specify priors in the
        original measurement units.
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
        "hazard_lambda": ParamDef(
            constraint=Interval(float, 1, None, Closed.LEFT),
            description="Expected run length (constant hazard = 1/lambda).",
        ),
        "mu": ParamDef(
            constraint=Interval(float, None, None),
            description="Prior mean of segment observations.",
        ),
        "kappa": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Strength of the prior mean (must be > 0).",
        ),
        "alpha": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Prior shape of inverse-gamma (must be > 0).",
        ),
        "beta": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Prior scale of inverse-gamma (must be > 0).",
        ),
        "truncate": ParamDef(
            constraint=Interval(int, None, 0, Closed.RIGHT),
            description="Legacy natural-log cutoff in marginal decoding.",
            nullable=True,
        ),
        "cp_prob_threshold": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Min posterior probability to accept a change point.",
        ),
        "min_distance": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum segment length, including the edge segments.",
        ),
        "max_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Maximum number of change points to return.",
            nullable=True,
        ),
        "multivariate_strategy": ParamDef(
            constraint=StrOptions({"native", "l2", "ensembling"}),
            description="Strategy for multivariate data.",
        ),
        "tolerance": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Tolerance for aggregating CPs in ensembling.",
        ),
        "decoding": ParamDef(
            constraint=StrOptions({"map", "marginal"}),
            description="Joint MAP partition or thresholded boundary marginals.",
        ),
        "max_candidates": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Maximum interior candidates; None uses every position.",
            nullable=True,
        ),
        "standardize": ParamDef(
            constraint=HasType((bool,)),
            description="Center and scale each channel before inference.",
        ),
        "_cross_constraints": [
            DataDependent(
                "min_distance < n_samples",
                "min_distance must be smaller than the series length",
            ),
        ],
    }

    def __init__(
        self,
        hazard_lambda: float = 1000,
        mu: float = 0.0,
        kappa: float = 1.0,
        alpha: float = 1.0,
        beta: float = 10.0,
        truncate: int | None = None,
        cp_prob_threshold: float = 0.5,
        min_distance: int = 25,
        max_cps: int | None = None,
        multivariate_strategy: str = "native",
        tolerance: int | float = 0,
        axis: int = 0,
        *,
        decoding: str = "map",
        max_candidates: int | None = 2000,
        standardize: bool = True,
    ):
        self.hazard_lambda = float(hazard_lambda)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.truncate = None if truncate is None else int(truncate)
        self.cp_prob_threshold = float(cp_prob_threshold)
        self.min_distance = int(max(min_distance, 1))
        self.max_cps = max_cps if max_cps is None else int(max_cps)
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        self.decoding = decoding
        self.max_candidates = max_candidates
        self.standardize = standardize
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

        if n_samples < 2 or self.max_cps == 0:
            return np.empty(0, dtype=np.int64)

        # Reduce first in the L2 mode: centering individual channels first
        # can turn a pure magnitude step into two vectors of equal norm.
        if n_channels > 1 and self.multivariate_strategy == "l2":
            data = multivariate_l2_norm(data)[:, None]
            n_channels = 1
        if self.standardize:
            centered = data - data[0]
            centered = centered - centered.mean(axis=0)
            scale = centered.std(axis=0)
            data = centered / np.where(scale == 0, 1.0, scale)

        if n_channels > 1 and self.multivariate_strategy == "ensembling":
            return self._predict_ensembling(data)

        return self._run_bocd(data, n_samples)

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
        candidates = aggregate_change_points(
            all_cps,
            n_cp=n_cp,
            tolerance=self.tolerance,
            signal_len=n_samples,
        )
        # Per-channel spacing does not by itself space the union of channels.
        selected: list[int] = []
        for point in candidates:
            if not selected or point - selected[-1] >= self.min_distance:
                selected.append(int(point))
        return np.asarray(selected, dtype=np.int64)

    def _run_bocd(self, signal: np.ndarray, n_samples: int) -> np.ndarray:
        """Decode a Gaussian product partition of a scalar or vector signal."""
        hazard_prob = float(
            np.clip(
                1.0 / self.hazard_lambda,
                np.nextafter(0.0, 1.0),
                np.nextafter(1.0, 0.0),
            )
        )
        segments = _GaussianSegments(
            signal,
            mu0=self.mu,
            kappa0=self.kappa,
            alpha0=self.alpha,
            beta0=self.beta,
            max_candidates=self.max_candidates,
        )
        if self.decoding == "map":
            return _map_partition(
                segments, hazard_prob, self.min_distance, self.max_cps
            )
        cp_probabilities = _boundary_marginals(
            segments, hazard_prob, self.min_distance, self.truncate
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
        return {"min_distance": 20}


class _StudentTLogMarginal:
    """Normal-Gamma marginal likelihood for a univariate Gaussian segment.

    The detector evaluates vectors of segments from sufficient statistics.
    The scalar ``pdf`` interface is retained for the legacy inference helper.
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

    def _prepare(self, n_samples: int) -> None:
        """Cache only the O(n) terms that depend on segment length."""
        counts = np.arange(n_samples + 1)
        kappa_n = self.kappa0 + counts
        self._alpha_n = self.alpha0 + 0.5 * counts
        self._mean_weights = self.kappa0 * counts / (2 * kappa_n)
        self._log_constants = (
            gammaln(self._alpha_n)
            - gammaln(self.alpha0)
            + self.alpha0 * np.log(self.beta0)
            + 0.5 * (np.log(self.kappa0) - np.log(kappa_n))
            - 0.5 * counts * np.log(2 * np.pi)
        )

    def _log_marginals(
        self,
        mean_deviations: np.ndarray,
        sample_ss: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """Evaluate segment evidence from stable running sample moments."""
        beta_n = (
            self.beta0
            + 0.5 * sample_ss
            + self._mean_weights[counts] * mean_deviations**2
        )
        return self._log_constants[counts] - self._alpha_n[counts] * np.log(beta_n)

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
            - 0.5 * n * np.log(2 * np.pi)
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

    peaks = find_peaks(np.r_[0.0, probabilities, 0.0])[0] - 1
    order = peaks[np.argsort(-probabilities[peaks], kind="stable")]
    selected: list[int] = []
    for idx in order:
        # Checked before selecting: ``max_cps=0`` selects no change point.
        if max_cps is not None and len(selected) >= max_cps:
            break
        prob = probabilities[idx]
        if prob < threshold:
            break

        cp_position = idx + 1  # boundary is after sample idx
        if cp_position < min_distance or cp_position > n_samples - min_distance:
            continue

        if (
            selected
            and np.min(np.abs(np.asarray(selected) - cp_position)) < min_distance
        ):
            continue

        selected.append(cp_position)

    selected.sort()
    return np.asarray(selected, dtype=np.int64)
