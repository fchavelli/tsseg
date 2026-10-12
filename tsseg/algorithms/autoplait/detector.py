"""
This module provides an aeon-compatible wrapper for the AutoPlait algorithm.
"""

import warnings

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    Interval,
    ParamDef,
)
from .autoplait_c import AutoPlait


class AutoPlaitDetector(BaseSegmenter):
    """
    Wrapper for the AutoPlait state detection algorithm.

    Parameters
    ----------
    n_cps : int or None, default=None
        Deprecated and ignored. AutoPlait chooses the number of segments and
        regimes itself and the bundled binary has no guided mode, so this
        value has never changed the output. Passing it emits a
        ``FutureWarning`` at ``fit``; it will be removed in a future release.

    Notes
    -----
    The label of a time point is the number of the AutoPlait regime its
    segment belongs to: segments of the same regime share a label, and each
    change point is the first index of a segment.

    References
    ----------
    .. [1] Y. Matsubara, Y. Sakurai, and C. Faloutsos, "AutoPlait: automatic mining of
       co-evolving time sequences," SIGMOD, 2014, doi: 10.1145/2588555.2588556.
    .. [2] Online Semantic Segmentation project page,
       https://sites.google.com/site/onlinesemanticsegmentation/ (reference implementation).
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    _parameter_schema = {
        "n_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Deprecated and ignored: the binary has no guided mode.",
            nullable=True,
        ),
    }

    def __init__(self, n_cps=None):
        super().__init__(axis=0)
        self.n_cps = n_cps

    def _fit(self, X, y=None):
        """Warn about the deprecated ``n_cps``; AutoPlait runs in ``predict``.

        Parameters
        ----------
        X : np.ndarray
            The time series, of shape (n_timepoints, n_channels).
        y : None
            Not used: AutoPlait is unsupervised.

        Returns
        -------
        self
        """
        if self.n_cps is not None:
            warnings.warn(
                f"{type(self).__name__}: `n_cps` is deprecated and ignored. "
                "AutoPlait chooses the number of segments itself (the binary "
                "has no guided mode); `n_cps` will be removed in a future "
                "release.",
                FutureWarning,
                stacklevel=3,
            )
        return self

    def _predict(self, X, y=None):
        """
        Segment the time series and assign a regime label to each time point.

        This method calls the underlying AutoPlait C implementation via the
        `AutoPlait` runner class.

        Parameters
        ----------
        X : np.ndarray
            The time series to segment, of shape (n_timepoints, n_channels).
        y : None
            Not used, for API compatibility.

        Returns
        -------
        np.ndarray
            The regime label of each time point, of shape (n_timepoints,).
        """
        ts = np.asarray(X)
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]
        elif ts.ndim > 2:
            ts = ts.reshape(ts.shape[0], -1)

        return AutoPlait()._run_autoplait(ts, None)
