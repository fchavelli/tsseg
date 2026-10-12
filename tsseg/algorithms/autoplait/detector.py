"""
This module provides an aeon-compatible wrapper for the AutoPlait algorithm.
"""

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

    Notes
    -----
    The label of a time point is the number of the AutoPlait regime its
    segment belongs to: segments of the same regime share a label, and each
    change point is the first index of a segment.

    The bundled binary takes no number of segments: ``n_cps`` is passed to it,
    rejected, and the binary then runs unguided, so the number of change
    points it returns does not depend on ``n_cps``.

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
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "n_cps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of change points (required if y not provided).",
            nullable=True,
        ),
    }

    def __init__(self, n_cps=None):
        super().__init__(axis=0)
        self.n_cps = n_cps

    def _fit(self, X, y):
        """
        Fit the segmenter.
        If y is provided (semi-supervised), we extract n_cps from it.
        If y is None (unsupervised), we rely on the __init__ n_cps or raise error.
        """
        if y is None and self.n_cps is None:
            raise ValueError(
                f"{self.__class__.__name__} requires `n_cps` or `y` to be provided. "
                "It is not capable of pure unsupervised learning without hints."
            )

        # If unsupervised and capable, we keep self.n_cps as defined in __init__ (or None if algo handles it)
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

        return AutoPlait()._run_autoplait(ts, self.n_cps)
