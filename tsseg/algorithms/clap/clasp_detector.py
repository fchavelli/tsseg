"""
This module provides a wrapper for the ClaSP algorithm to integrate it
with the tsseg library's API.
"""

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)
from .segmentation import BinaryClaSPSegmentation


class ClaspDetector(BaseSegmenter):
    """
    A wrapper for the ClaSP (Classification Score Profile) algorithm for
    time series segmentation, using binary segmentation, compatible with aeon.

    Parameters
    ----------
    n_segments : str or int, default="learn"
        The number of segments to split the time series into. If "learn", the
        number is inferred automatically: each change point must pass the
        ``validation``. An integer is a given number (guided mode), returned
        without validation. ``1`` returns no change point.
    n_change_points : int, optional
        The number of change points (guided mode). Takes precedence over
        ``n_segments`` when set; ``0`` returns no change point.
    n_estimators : int, default=10
        The number of ClaSPs in the ensemble.
    window_size : str or int, default="suss"
        The window size or the method to determine it ('suss', 'fft', 'acf').
    k_neighbours : int, default=3
        The number of nearest neighbors to use in the ClaSP algorithm.
    excl_radius : int, default=5
        The exclusion radius, in multiples of the window size.
    distance : str, default="znormed_euclidean_distance"
        The distance used to find the k nearest neighbours:
        ``"znormed_euclidean_distance"``, ``"euclidean_distance"`` or
        ``"cinvariant_euclidean_distance"`` (complexity invariant).
    score : str, default="roc_auc"
        The classification score of the profile: ``"roc_auc"`` or ``"f1"``.
    early_stopping : bool, default=True
        Stop adding ClaSPs to the ensemble once a validated change point is
        found or a new ClaSP does not improve the best score. In the guided
        mode, where nothing is validated, the ensemble stops after its first
        ClaSP, fitted on the whole segment.
    validation : str, default="significance_test"
        How a candidate change point is accepted when the number of segments
        is learned: ``"significance_test"`` (statistical test on the score
        profile) or ``"score_threshold"`` (minimum score). Not used in the
        guided mode.
    threshold : str or float, default="default"
        The threshold of the validation: a p-value for
        ``"significance_test"``, a minimum score for ``"score_threshold"``.
        ``"default"`` means ``1e-15`` (univariate) or ``1e-30``
        (multivariate) for the test, and ``0.75`` for the score.
    n_jobs : int, default=-1
        The number of numba threads. Values below 1 mean all the threads
        numba may use; larger values are capped at that number.
    axis : int, default=0
        The time axis of ``X``: ``0`` if time points are rows.

    Attributes
    ----------
    change_points_ : np.ndarray of int
        The indices of the detected change points, sorted in ascending order.
        Available only after calling the `fit` or `fit_predict` method.

    Notes
    -----
    In the guided mode (``n_change_points``, or an integer ``n_segments``),
    the count is returned as it is: the candidate change points are not
    validated, so ``n_change_points=k`` gives ``k`` change points, fewer only
    when the segments are too short to split further (each part must span at
    least ``excl_radius`` windows). claspy validates them in this mode too, so
    it may return fewer, or none.

    The ensemble draws its temporal constraints with a fixed seed (2357, as in
    claspy).
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "change_point_detection",
        "python_dependencies": ["numba"],
        "fit_is_empty": False,
        "returns_dense": True,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "n_segments": ParamDef(
            constraint=[
                StrOptions({"learn"}),
                Interval(int, 1, None, Closed.LEFT),
            ],
            description="Number of segments ('learn' = auto-detect).",
        ),
        "n_change_points": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points (injected by supervision pipeline).",
            nullable=True,
            ui_hidden=True,
        ),
        "n_estimators": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of ClaSPs in the ensemble.",
        ),
        "window_size": ParamDef(
            constraint=[
                StrOptions({"suss", "fft", "acf"}),
                Interval(int, 1, None, Closed.LEFT),
            ],
            description="Window size or auto-detection method.",
        ),
        "k_neighbours": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of nearest neighbours for ClaSP.",
        ),
        "excl_radius": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Exclusion radius (multiples of window size).",
        ),
        "distance": ParamDef(
            constraint=StrOptions(
                {
                    "znormed_euclidean_distance",
                    "euclidean_distance",
                    "cinvariant_euclidean_distance",
                }
            ),
            description="Distance function for k-NN.",
        ),
        "score": ParamDef(
            constraint=StrOptions({"roc_auc", "f1"}),
            description="Scoring function for the profile ('roc_auc' or 'f1').",
        ),
        "early_stopping": ParamDef(
            constraint=HasType((bool,)),
            description="Stop early when no significant split is found.",
        ),
        "validation": ParamDef(
            constraint=StrOptions({"significance_test", "score_threshold"}),
            description="Validation method for change point significance.",
        ),
        "threshold": ParamDef(
            constraint=[
                StrOptions({"default"}),
                Interval(float, 0, 1, Closed.BOTH),
            ],
            description="Threshold for validation ('default' or float in [0,1]).",
        ),
        "n_jobs": ParamDef(
            constraint=Interval(int, -1, None, Closed.LEFT),
            description="Number of parallel jobs (-1 = all CPUs).",
        ),
        "_cross_constraints": [
            DataDependent(
                "window_size < n_samples or isinstance(window_size, str)",
                "Integer window_size must be < series length.",
            ),
        ],
    }

    def __init__(
        self,
        n_segments="learn",
        n_change_points=None,
        n_estimators=10,
        window_size="suss",
        k_neighbours=3,
        excl_radius=5,
        distance="znormed_euclidean_distance",
        score="roc_auc",
        early_stopping=True,
        validation="significance_test",
        threshold="default",
        n_jobs=-1,
        axis=0,
    ):
        self.n_segments = n_segments
        self.n_change_points = n_change_points  # Injected by the supervision pipeline
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
        effective_n_segments = self.n_segments
        if self.n_change_points is not None:
            effective_n_segments = int(self.n_change_points) + 1
        # A given number of segments is returned as it is: only a learned
        # number is validated (claspy validates both).
        validation = self.validation if effective_n_segments == "learn" else None

        self.model = BinaryClaSPSegmentation(
            n_segments=effective_n_segments,
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            excl_radius=self.excl_radius,
            distance=self.distance,
            score=self.score,
            early_stopping=self.early_stopping,
            validation=validation,
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
