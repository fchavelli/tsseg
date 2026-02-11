"""
This module provides an aeon-compatible wrapper for the AutoPlait algorithm.
"""
import numpy as np
from ..base import BaseSegmenter
from .autoplait_c import AutoPlait

class AutoPlaitDetector(BaseSegmenter):
    """
    Wrapper for the AutoPlait state detection algorithm.

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
        Segment the time series to find change points and assign state labels.

        This method calls the underlying AutoPlait C implementation via the
        `AutoPlait` runner class.

        Parameters
        ----------
        X : np.ndarray
            The time series to segment of shape (n_channels, n_timepoints).
        y : None
            Not used, for API compatibility.

        Returns
        -------
        np.ndarray
            The predicted state labels as a 1D array of shape (n_timepoints,).
        """
        # Ensure a 2D array where rows are time points and columns are features.
        ts = np.asarray(X)
        if ts.ndim == 1:
            ts2d = ts[:, np.newaxis]
        elif ts.ndim == 2:
            ts2d = ts
        else:
            ts2d = ts.reshape(ts.shape[0], -1)

        # 1. Instantiate the runner class from autoplait.py
        autoplait_runner = AutoPlait()

        # 2. Run the C code to get changepoints and labels
        change_points, segment_labels = autoplait_runner._run_autoplait(
            ts2d, self.n_cps
        )

        change_points = np.asarray(change_points, dtype=int)
        segment_labels = np.asarray(segment_labels, dtype=int)

        n_samples = ts2d.shape[0]
        if segment_labels.size == 0:
            if change_points.size == 0:
                return np.zeros(n_samples, dtype=int)
            segment_labels = np.arange(change_points.size + 1, dtype=int)

        if change_points.size > 0:
            change_points = np.sort(change_points)
            change_points = change_points[(change_points > 0) & (change_points < n_samples)]

        expected_segments = change_points.size + 1
        if segment_labels.size < expected_segments:
            pad_value = segment_labels[-1]
            pad_width = expected_segments - segment_labels.size
            segment_labels = np.pad(
                segment_labels, (0, pad_width), mode="constant", constant_values=pad_value
            )
        elif segment_labels.size > expected_segments:
            segment_labels = segment_labels[:expected_segments]

        dense_labels = np.empty(n_samples, dtype=int)
        boundaries = list(change_points.tolist())
        boundaries.append(n_samples)

        start = 0
        for label, end in zip(segment_labels, boundaries):
            end_idx = min(max(end, start), n_samples)
            dense_labels[start:end_idx] = label
            start = end_idx

        if start < n_samples:
            dense_labels[start:] = segment_labels[-1]

        return dense_labels
