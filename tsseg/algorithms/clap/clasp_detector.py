"""
This module provides a wrapper for the ClaSP algorithm to integrate it
with the tsseg library's API.
"""
import numpy as np
from ..base import BaseSegmenter
from .segmentation import BinaryClaSPSegmentation

class ClaspDetector(BaseSegmenter):
    """
    A wrapper for the ClaSP (Classification Score Profile) algorithm for
    time series segmentation, using binary segmentation, compatible with aeon.

    Parameters
    ----------
    n_segments : str or int, default="learn"
        The number of segments to split the time series into. If "learn", the
        number is inferred automatically.
    n_estimators : int, default=10
        The number of ClaSPs in the ensemble.
    window_size : str or int, default="suss"
        The window size or the method to determine it ('suss', 'fft', 'acf').
    k_neighbours : int, default=3
        The number of nearest neighbors to use in the ClaSP algorithm.
    excl_radius : int, default=5
        The exclusion radius, in multiples of the window size.

    Attributes
    ----------
    change_points_ : list of int
        The indices of the detected change points, sorted in ascending order.
        Available only after calling the `fit` or `fit_predict` method.
    """
    _tags = {
        "capability:unequal_length": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "change_point_detection",
        "fit_is_empty": False,
        "returns_dense": True,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(self, n_segments="learn", n_change_points=None, n_estimators=10, window_size="suss", k_neighbours=3, excl_radius=5, distance="znormed_euclidean_distance", score="roc_auc",
                 early_stopping=True, validation="significance_test", threshold="default", n_jobs=-1, axis=0):
        self.n_segments = n_segments
        self.n_change_points = n_change_points # Injected by the supervision pipeline
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.excl_radius = excl_radius
        self.distance = distance
        self.score = score
        self.early_stopping = early_stopping
        self.validation = validation
        self.threshold = threshold
        self.n_jobs = n_jobs
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        """
        Fits the ClaSP model to the provided time series data.
        """
        # Extra logic to handle n_change_points if provided.
        # For some reason (to investigate later), providing self.n_segments directly
        # to BinaryClaSPSegmentation does not work when it is "learn".
        # Edit: issue was inconsistency with base class, solved.
        effective_n_segments = self.n_segments
        if self.n_change_points is not None:
            effective_n_segments = int(self.n_change_points) + 1

        self.model = BinaryClaSPSegmentation(
            n_segments=effective_n_segments,
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            excl_radius=self.excl_radius,
            distance=self.distance,
            score=self.score,
            early_stopping=self.early_stopping,
            validation=self.validation,
            threshold=self.threshold,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X)
        if X.ndim == 1:
            self.ts_len_ = X.shape[0]
        else:
            self.ts_len_ = X.shape[0] if self.axis == 0 else X.shape[1]
        self.change_points_ = self.model.predict()
        # Ensure change points are integers
        self.change_points_ = np.array(self.change_points_, dtype=int)
        self._is_fitted = True
        return self

    def _predict(self, X, y=None):
        """
        Predicts the change points and returns them.
        """
        # The predict method of BinaryClaSPSegmentation returns the change points.
        self._check_is_fitted()
        
        return self.change_points_