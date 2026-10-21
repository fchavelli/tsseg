"""
This module provides a wrapper for the CLaP algorithm to integrate it
with the tsseg library's API.
"""

import numpy as np

from ..base import BaseSegmenter

# from ..clap.clasp_detector import ClaspDetector
from ..clap.segmentation import BinaryClaSPSegmentation
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)
from .clap import CLaP
from .utils import create_state_labels


class ClapDetector(BaseSegmenter):
    """
    A wrapper for the CLaP (Classification Label Profile)
    algorithm for time series state detection, compatible with aeon.

    Parameters
    ----------
    window_size : str or int, default="suss"
        The window size or the method to determine it ('suss', 'fft', 'acf').
    classifier : str, default="rocket"
        The aeon time series classifier that is cross-validated on the
        segments: ``"rocket"`` (``RocketClassifier``), ``"mrhydra"``
        (``MultiRocketHydraClassifier``), ``"weasel"`` (``WEASEL_V2``),
        ``"quant"`` (``QUANTClassifier``), ``"rdst"`` (``RDSTClassifier``),
        ``"proximityforest"`` (``ProximityForest``), ``"freshprince"``
        (``FreshPRINCEClassifier``) or ``"inception"``
        (``IndividualInceptionClassifier``, which needs the aeon deep learning
        extras, ``pip install aeon[dl]``).
    merge_score : str, default="cgain"
        The score that decides whether two confused segment labels are merged:
        ``"cgain"`` (classification gain), ``"f1_score"`` (macro F1),
        ``"log_loss"``, ``"ami"`` (adjusted mutual information),
        ``"hamming_loss"`` or ``"roc_auc"``.
    n_splits : int, default=5
        Number of splits for cross-validation.
    n_jobs : int, default=1
        The number of jobs to run in parallel. -1 means using all processors.
    sample_size : int, default=1000
        The number of samples to use for training the classifier.
    random_state : int, default=42
        Random seed for reproducibility.
    axis : int, default=0
        The axis of the time series data.
    n_change_points : int, optional
        The number of change points to find (guided mode). Takes precedence
        over ``n_segments``; ``0`` yields a single segment, hence one state.
    change_points : array-like of int, optional
        Known change points (exact mode). When given, ClaSP is not run and
        ``n_change_points`` and ``n_segments`` are ignored.
    n_segments : int, optional
        The number of segments to find (guided mode), passed to ClaSP.

    Notes
    -----
    CLaP labels the segments between change points. Three usage patterns are
    supported:

    * **Exact** – pass the known change points as ``change_points``; they are
      used as they are.
    * **Guided** – pass ``n_change_points`` (or ``n_segments``). The count is
      forwarded to ``BinaryClaSPSegmentation``, which finds that many change
      points.
    * **Fully unsupervised** – the default behaviour: ClaSP estimates the
      change points and their number.

    ``y`` is never used: :meth:`fit` ignores it.

    Attributes
    ----------
    states_ : np.ndarray
            An array of state labels for each time point.
            Available only after calling the `fit` or `fit_predict` method.
    change_points_ : np.ndarray
            The change points used for fitting the model.
    """

    _tags = {
        "capability:unequal_length": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        "python_dependencies": ["numba", "aeon"],
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "window_size": ParamDef(
            constraint=[
                StrOptions({"suss", "fft", "acf"}),
                Interval(int, 4, None, Closed.LEFT),
            ],
            description="Window size or auto-detection method.",
        ),
        "classifier": ParamDef(
            constraint=StrOptions(
                {
                    "mrhydra",
                    "rocket",
                    "weasel",
                    "quant",
                    "rdst",
                    "proximityforest",
                    "freshprince",
                    "inception",
                }
            ),
            description="aeon classifier cross-validated on the segments.",
        ),
        "merge_score": ParamDef(
            constraint=StrOptions(
                {"cgain", "f1_score", "log_loss", "ami", "hamming_loss", "roc_auc"}
            ),
            description="Scoring function for segment merging.",
        ),
        "n_splits": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="Number of cross-validation splits.",
        ),
        "n_jobs": ParamDef(
            constraint=Interval(int, -1, None, Closed.LEFT),
            description="Number of parallel jobs (-1 = all CPUs).",
        ),
        "sample_size": ParamDef(
            constraint=Interval(int, 10, None, Closed.LEFT),
            description="Samples for classifier training.",
        ),
        "random_state": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Random seed.",
            nullable=True,
        ),
        "n_change_points": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points (overrides n_segments).",
            nullable=True,
        ),
        "n_segments": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of segments to detect.",
            nullable=True,
        ),
        "change_points": ParamDef(
            constraint=HasType((list, tuple, np.ndarray)),
            description="Known change points (exact mode; ClaSP is not run).",
            nullable=True,
            ui_hidden=True,
        ),
        "_cross_constraints": [
            DataDependent(
                "window_size < n_samples or isinstance(window_size, str)",
                "Integer window_size must be < series length",
            ),
        ],
    }

    def __init__(
        self,
        window_size="suss",
        classifier="rocket",
        merge_score="cgain",
        n_splits=5,
        n_jobs=1,
        sample_size=1_000,
        random_state=42,
        axis=0,
        n_change_points=None,
        change_points=None,
        n_segments=None,
    ):
        self.window_size = window_size
        self.classifier = classifier
        self.merge_score = merge_score
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.sample_size = sample_size
        self.random_state = random_state
        self.n_change_points = n_change_points
        self.change_points = change_points
        self.n_segments = n_segments
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        """
        Fits the CLaP model to the provided time series data.

        The change points are ``change_points`` when given (exact mode);
        otherwise ``BinaryClaSPSegmentation`` finds them, guided by
        ``n_change_points`` or ``n_segments`` when either is set.

        Parameters
        ----------
        X : np.ndarray
            A 1D or 2D numpy array representing the time series.
        y : None
            Ignored.

        Returns
        -------
        self : ClapDetector
            The fitted instance of the detector.
        """
        self.ts_len_ = X.shape[0]

        self._clap = CLaP(
            window_size=self.window_size,
            classifier=self.classifier,
            merge_score=self.merge_score,
            n_splits=self.n_splits,
            n_jobs=self.n_jobs,
            sample_size=self.sample_size,
            random_state=self.random_state,
        )

        # Determine change points
        if self.change_points is not None:
            # Exact mode (via init). CLaP reads ``change_points.shape``.
            self.change_points_ = np.asarray(self.change_points, dtype=int)
        else:
            # Unsupervised or Guided Semi-Supervised
            target_n_segments = self.n_segments
            # Extra logic to handle n_change_points if provided.
            # For some reason (to investigate later), providing target_n_segments directly
            # to BinaryClaSPSegmentation does not work when it is "learn".
            # Edit: issue was inconsistency with base class, solved.
            if self.n_change_points is not None:
                target_n_segments = int(self.n_change_points) + 1

            # If target_n_segments is None, BinaryClaSPSegmentation uses "learn"
            n_seg_arg = target_n_segments if target_n_segments is not None else "learn"

            clasp = BinaryClaSPSegmentation(
                n_segments=n_seg_arg, n_jobs=self.n_jobs, window_size=self.window_size
            )
            clasp.fit_predict(X)
            self.change_points_ = clasp.change_points

        self._clap.fit(X, change_points=self.change_points_)

        # All computation is done in fit, pre-calculating the states.
        labels = self._clap.get_segment_labels()
        cps = self._clap.get_change_points()

        self.states_ = create_state_labels(cps, labels, self.ts_len_)

        return self

    def _predict(self, X):
        """
        Predicts the states for the given time series using the fitted model.

        Parameters
        ----------
        X : np.ndarray
            The input time series.
        y : np.ndarray, optional
            Not used, present here for compatibility with the method signature.

        Returns
        -------
        np.ndarray
            The predicted state labels.
        """
        # The states are pre-computed and stored in `fit`.
        return self.states_
