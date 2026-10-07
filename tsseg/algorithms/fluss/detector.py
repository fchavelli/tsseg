"""FLUSS (Fast Low-cost Unipotent Semantic Segmentation) Segmenter."""

__maintainer__ = []
__all__ = ["FLUSSDetector"]

import numpy as np
import pandas as pd

try:
    import stumpy

    _HAS_STUMPY = True
except ImportError:
    _HAS_STUMPY = False

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    Interval,
    ParamDef,
    StrOptions,
)
from ..utils import (
    aggregate_change_points,
    consensus_change_points,
    multivariate_l2_norm,
)


class FLUSSDetector(BaseSegmenter):
    """FLUSS (Fast Low-cost Unipotent Semantic Segmentation) Segmenter.

    FLOSS [1]_ FLUSS is a domain-agnostic online semantic segmentation method that
    operates on the assumption that a low number of arcs crossing a given index point
    indicates a high probability of a semantic change. Segments are called regimes in
    the original paper and stumpy package, but here we use the term segments to be
    consistent with state detection terminology.

    Parameters
    ----------
    window_size : int, default = 10
        Size of window for sliding, based on the period length of the data.
    n_segments : int or None, default = 2
        The number of segments to search (change points + 1). ``None``: the
        number is not known, and every minimum of the corrected arc curve below
        ``threshold`` is a change point.
    exclusion_factor : int, default 5
        The multiplying factor for the segment exclusion zone
    multivariate_strategy : str, default="ensembling"
        Strategy for handling multivariate data: "ensembling" or "l2".
    tolerance : int or float, default=0.01
        Tolerance for aggregating change points in ensembling strategy; a float
        below 1 is a fraction of the series length.
    threshold : float, default=0.5
        Used when ``n_segments`` is ``None``: the regime extraction of FLUSS
        takes the lowest point of the corrected arc curve (CAC, in [0, 1]) and
        masks its exclusion zone, as long as that point is below
        ``threshold`` (the online variant FLOSS flags a regime change the same
        way, when the CAC drops below a threshold).
    consensus : float, default=0.0
        Ensembling with ``n_segments=None``: the change points of the channels
        that lie within ``tolerance`` of each other are merged into one,
        returned if at least ``consensus`` of the channels (one at least)
        detected it.

    References
    ----------
    .. [1] Gharghabi S, Ding Y, Yeh C-CM, Kamgar K, Ulanova L, Keogh E. Matrix
       Profile VIII: Domain Agnostic Online Semantic Segmentation at Superhuman
       Performance Levels. In: 2017 IEEE International Conference on Data Mining
       (ICDM). IEEE; 2017. p. 117-26.

    Examples
    --------
    >>> from aeon.segmentation import FLUSSSegmenter
    >>> from aeon.datasets import load_gun_point_segmentation
    >>> X, true_period_size, cps = load_gun_point_segmentation()
    >>> fluss = FLUSSSegmenter(window_size=10, n_segments=2)  # doctest: +SKIP
    >>> found_cps = fluss.fit_predict(X)  # doctest: +SKIP
    >>> profiles = fluss.profiles  # doctest: +SKIP
    >>> scores = fluss.scores  # doctest: +SKIP
    """

    _tags = {
        "capability:univariate": True,
        "fit_is_empty": False,
        "capability:multivariate": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "window_size": ParamDef(
            constraint=Interval(int, 3, None, Closed.LEFT),
            description="Sliding window size (based on period length).",
        ),
        "n_segments": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of segments to find (None: CAC threshold).",
            nullable=True,
        ),
        "exclusion_factor": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Multiplier for segment exclusion zone.",
        ),
        "multivariate_strategy": ParamDef(
            constraint=StrOptions({"ensembling", "l2"}),
            description="Strategy for multivariate data.",
        ),
        "tolerance": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Tolerance for CP aggregation in ensembling.",
        ),
        "threshold": ParamDef(
            constraint=Interval(float, 0, 1, Closed.BOTH),
            description="CAC threshold of a change point when n_segments is None.",
        ),
        "consensus": ParamDef(
            constraint=Interval(float, 0, 1, Closed.BOTH),
            description="Fraction of channels that must agree (ensembling, no n_segments).",
        ),
        "_cross_constraints": [
            DataDependent(
                "window_size < n_samples",
                "window_size must be less than the series length",
            ),
            DataDependent(
                "n_segments <= n_samples // window_size",
                "Too many segments for the given series length and window size",
            ),
        ],
    }

    def __init__(
        self,
        window_size=10,
        n_segments=2,
        exclusion_factor=5,
        axis=0,
        multivariate_strategy="ensembling",
        tolerance=0.01,
        threshold=0.5,
        consensus=0.0,
    ):
        if not _HAS_STUMPY:
            raise ImportError(
                "stumpy is not installed. Please install it to use FLUSSDetector."
            )
        self.window_size = window_size
        self.n_segments = n_segments
        self.exclusion_factor = exclusion_factor
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        self.threshold = threshold
        self.consensus = consensus
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        return self

    def _predict(self, X: np.ndarray):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : np.ndarray
            1D time series to be segmented.

        Returns
        -------
        list
            List of change points found in X.
        """
        if self.n_segments is not None and self.n_segments < 1:
            raise ValueError(
                "The number of regimes must be set to an integer greater than or equal to 1"
            )

        # Short-circuit for 1 segment (no change points)
        if self.n_segments == 1:
            self.found_cps = []
            self.scores = np.array([])
            self.profile = None
            return self.found_cps

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]

        signal_len, dim = X.shape

        if dim > 1 and self.multivariate_strategy == "ensembling":
            all_detected_indices = []
            per_channel = []
            # Ensembling: run FLUSS on each dimension
            for d in range(dim):
                dim_values = X[:, d]
                cps, _, _ = self._run_fluss(dim_values)
                all_detected_indices.extend(cps)
                per_channel.append(cps)

            # Aggregate results
            # Note: n_segments = n_changepoints + 1 (roughly), but FLUSS returns indices.
            # We want n_segments - 1 change points.
            if self.n_segments is None:
                self.found_cps = consensus_change_points(
                    per_channel, self._tolerance_samples(signal_len), self.consensus
                )
            else:
                n_cp = self.n_segments - 1
                self.found_cps = aggregate_change_points(
                    all_detected_indices, n_cp, self.tolerance, signal_len=signal_len
                )

            # For profiles/scores in multivariate case, it's ambiguous.
            # We could average them, but for now we leave them as None or last dimension.
            self.profiles = None
            self.scores = None

        else:
            # Univariate or L2 strategy
            if dim > 1:
                # L2 strategy
                signal = multivariate_l2_norm(X)
            else:
                signal = X.flatten()

            self.found_cps, self.profiles, self.scores = self._run_fluss(signal)

        return self.found_cps

    def predict_scores(self, X):
        """Return scores in FLUSS's profile for each annotation.

        Parameters
        ----------
        np.ndarray
            1D time series to be segmented.

        Returns
        -------
        np.ndarray
            Scores for sequence X
        """
        # Force predict to run to populate scores if possible
        self._predict(X)
        return self.scores

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {"profile": self.profile}

    def _tolerance_samples(self, signal_len: int) -> int:
        if isinstance(self.tolerance, float) and self.tolerance < 1.0:
            return int(self.tolerance * signal_len)
        return int(self.tolerance)

    def _run_fluss(self, X):
        mp = stumpy.stump(X, m=self.window_size)
        if self.n_segments is None:
            # Corrected arc curve only; the regimes are read off with the threshold.
            self.profile, _ = stumpy.fluss(
                mp[:, 1],
                L=self.window_size,
                excl_factor=self.exclusion_factor,
                n_regimes=1,
            )
            self.found_cps = self._threshold_regimes(self.profile)
        else:
            self.profile, self.found_cps = stumpy.fluss(
                mp[:, 1],
                L=self.window_size,
                excl_factor=self.exclusion_factor,
                n_regimes=self.n_segments,  # regimes -> segments for consistency with state detection naming
            )
        self.scores = self.profile[self.found_cps]

        return self.found_cps, self.profile, self.scores

    def _threshold_regimes(self, cac: np.ndarray) -> np.ndarray:
        """Regime extraction (REA) of FLUSS, stopped by ``threshold``.

        As ``stumpy.fluss`` with a known number of regimes: take the lowest
        point of the corrected arc curve, mask ``exclusion_factor *
        window_size`` samples on each side, repeat; here, while the lowest
        point is below ``threshold`` instead of a fixed number of times.
        """
        cac = np.array(cac, dtype=float, copy=True)
        zone = self.exclusion_factor * self.window_size
        locs = []
        while cac.size:
            loc = int(np.argmin(cac))
            if cac[loc] >= self.threshold:
                break
            locs.append(loc)
            cac[max(loc - zone, 0) : min(loc + zone, cac.shape[0])] = 1.0
        return np.asarray(sorted(locs), dtype=np.int64)

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
            Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps]
            The found change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change points

        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
        """
        return {"window_size": 5, "n_segments": 2}
