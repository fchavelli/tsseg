"""BinSeg (Binary segmentation) detector (legacy aeon-style wrapper).

.. deprecated:: 0.1.0
    Use :class:`tsseg.algorithms.binseg.detector.BinSegDetector` instead. This
    module will be removed in a future release once existing experiments migrate
    to the new implementation.
"""

from __future__ import annotations

import warnings

__maintainer__ = []
__all__ = ["BinSegDetector"]

import numpy as np
import pandas as pd

from ..ruptures.detection.binseg import Binseg
from ..base import BaseSegmenter

warnings.warn(
    "tsseg.algorithms.binseg.detector_aeon is deprecated; use"
    " tsseg.algorithms.binseg.detector.BinSegDetector instead.",
    DeprecationWarning,
    stacklevel=2,
)


class BinSegDetector(BaseSegmenter):
    """Binary segmentation change-point detector backed by ruptures' Binseg.

    The detector recursively locates one change point per iteration, splits the
    signal, and continues on each sub-signal until `n_cps` change points have
    been returned or no further splits are possible.

    Parameters
    ----------
    n_cps : int or None, default=None
        Maximum number of change points to detect. If None, ruptures decides.
    model : str, default="l2"
        Cost model passed to ruptures (e.g., "l1", "l2", "rbf").
    min_size : int, default=2
        Minimum segment length supplied to ruptures.
    jump : int, default=5
        Subsampling factor supplied to ruptures.

    References
    ----------
    .. [1] Bai, J. (1997). Estimating multiple breaks one at a time.
       Econometric Theory, 13(3), 315–352.
    .. [2] Fryzlewicz, P. (2014). Wild binary segmentation for multiple change-point
       detection. The Annals of Statistics, 42(6), 2243–2281.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms.binseg.detector import BinSegDetector
    >>> X = np.sin(np.linspace(0, np.pi, 50))
    >>> detector = BinSegDetector(n_cps=1)
    >>> cps = detector.fit_predict(X)  # doctest: +SKIP
    """

    _tags = {
        "fit_is_empty": True,
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "semi_supervised": False,
    }

    def __init__(self, n_cps=None, model="l2", min_size=2, jump=5):
        self.n_cps = n_cps
        self.model = model
        self.min_size = min_size
        self.jump = jump
        super().__init__(axis=0)

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
        X = X.squeeze()
        found_cps = self._run_binseg(X)
        return found_cps

    def get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {}

    def _run_binseg(self, X):
        binseg = Binseg(
            model=self.model, min_size=self.min_size, jump=self.jump
        ).fit(X)
        found_cps = np.array(binseg.predict(n_bkps=self.n_cps)[:-1], dtype=np.int64)

        return found_cps

    def _get_interval_series(self, X, found_cps):
        """Get the segmentation results based on the found change points.

        Parameters
        ----------
        X :         array-like, shape = [n]
           Univariate time-series data to be segmented.
        found_cps : array-like, shape = [n_cps] The found change points found

        Returns
        -------
        IntervalIndex:
            Segmentation based on found change points

        """
        cps = np.array(found_cps)
        start = np.insert(cps, 0, 0)
        end = np.append(cps, len(X))
        return pd.IntervalIndex.from_arrays(start, end)

    # @classmethod
    # def _get_test_params(cls, parameter_set="default"):
    #     """Return testing parameter settings for the estimator.

    #     Parameters
    #     ----------
    #     parameter_set : str, default="default"
    #         Name of the set of test parameters to return, for use in tests. If no
    #         special parameters are defined for a value, will return `"default"` set.

    #     Returns
    #     -------
    #     params : dict or list of dict, default = {}
    #         Parameters to create testing instances of the class
    #         Each dict are parameters to construct an "interesting" test instance, i.e.,
    #         `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
    #     """
    #     return {"n_cps": 1}