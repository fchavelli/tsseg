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
from ..utils import multivariate_l2_norm, aggregate_change_points

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
    n_segments : int, default = 2
        The number of segments to search (equal to change points - 1).
    exclusion_factor : int, default 5
        The multiplying factor for the segment exclusion zone
    multivariate_strategy : str, default="ensembling"
        Strategy for handling multivariate data: "ensembling" or "l2".
    tolerance : int, default=0
        Tolerance for aggregating change points in ensembling strategy.

    References
    ----------
    .. [1] Gharghabi S, Ding Y, Yeh C-CM, Kamgar K, Ulanova L, Keogh E. Matrix
    Profile VIII: Domain Agnostic Online Semantic Segmentation at Superhuman Performance
    Levels. In: 2017 IEEE International Conference on Data Mining (ICDM). IEEE; 2017.
    p. 117-26.

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

    def __init__(self, window_size=10, n_segments=2, exclusion_factor=5, axis=0, multivariate_strategy="ensembling", tolerance=0.01):
        if not _HAS_STUMPY:
            raise ImportError("stumpy is not installed. Please install it to use FLUSSDetector.")
        self.window_size = window_size
        self.n_segments = n_segments
        self.exclusion_factor = exclusion_factor
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
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
        if self.n_segments < 1:
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
            # Ensembling: run FLUSS on each dimension
            for d in range(dim):
                dim_values = X[:, d]
                cps, _, _ = self._run_fluss(dim_values)
                all_detected_indices.extend(cps)
            
            # Aggregate results
            # Note: n_segments = n_changepoints + 1 (roughly), but FLUSS returns indices.
            # We want n_segments - 1 change points.
            n_cp = self.n_segments - 1
            self.found_cps = aggregate_change_points(all_detected_indices, n_cp, self.tolerance, signal_len=signal_len)
            
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

    def _run_fluss(self, X):

        mp = stumpy.stump(X, m=self.window_size)
        self.profile, self.found_cps = stumpy.fluss(
            mp[:, 1],
            L=self.window_size,
            excl_factor=self.exclusion_factor,
            n_regimes=self.n_segments, # regimes -> segments for consistency with state detection naming
        )
        self.scores = self.profile[self.found_cps]

        return self.found_cps, self.profile, self.scores

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
