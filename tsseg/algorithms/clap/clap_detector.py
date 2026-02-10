"""
This module provides a wrapper for the CLaP algorithm to integrate it
with the tsseg library's API.
"""
from ..base import BaseSegmenter
from .clap import CLaP
from .utils import create_state_labels, extract_cps
# from ..clap.clasp_detector import ClaspDetector
from ..clap.segmentation import BinaryClaSPSegmentation

class ClapDetector(BaseSegmenter):
    """
    A wrapper for the CLaP (Classification Label Profile) 
    algorithm for time series state detection, compatible with aeon.

    Parameters
    ----------
    window_size : str or int, default="suss"
        The window size or the method to determine it ('suss', 'fft', 'acf').
    classifier : str, default="rocket"
        The classifier to use for state detection.
    merge_score : str, default="cgain"
        The scoring function to decide on merging segments.
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
    change_points : np.ndarray, optional
        Pre-computed change points. Used if `semi_supervised` is True.
    n_segments : int, optional
        The number of segments to find. Used if `semi_supervised` is False to
        guide the unsupervised change point detection.
    semi_supervised : bool, default=False
        If True, uses provided change points for fitting. Change points can be
        provided via the `change_points` parameter at initialization or via `y`
        in the `fit` method. If False, detects change points automatically.

    Notes
    -----
    The detector requires a list of change points to segment the series. Three
    usage patterns are supported:

    * **Exact semi-supervised** – enable ``semi_supervised=True`` and supply
        the true change points via the ``change_points`` parameter at
        initialisation or as ``y`` during :meth:`fit`. The semi-supervised tag is
        set accordingly so the test suite can forward labels.
    * **Guided semi-supervised** – keep ``semi_supervised=False`` but pass
        ``n_segments``. The provided segment count is forwarded to the underlying
        change-point estimator (``BinaryClaSPSegmentation`` by default), acting as
        a coarse constraint that typically sharpens the predicted breakpoints.
    * **Fully unsupervised** – the default behaviour, where no prior change
        point information is given. The change-point detector estimates breakpoints
        without guidance.

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
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(self, window_size="suss", classifier="rocket", merge_score="cgain", n_splits=5, n_jobs=1,
                 sample_size=1_000, random_state=42, axis=0, n_change_points=None, change_points=None, n_segments=None):
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

        In semi-supervised mode (`semi_supervised=True`), the model uses pre-defined
        change points. These can be provided either via the `change_points` parameter
        during initialization or via the `y` parameter to this method. If both are
        provided, `change_points` from initialization takes precedence.

        In unsupervised mode (`semi_supervised=False`), change points are detected
        automatically using `ClaspDetector`.

        Parameters
        ----------
        X : np.ndarray
            A 1D or 2D numpy array representing the time series.
        y : np.ndarray, optional
            Change points or state labels, used only if `semi_supervised` is True
            and the `change_points` parameter was not provided at initialization.

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
            random_state=self.random_state
        )

        # Determine change points
        if self.change_points is not None:
            # Exact mode (via init)
            self.change_points_ = self.change_points
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
                n_segments=n_seg_arg,
                n_jobs=self.n_jobs,
                window_size=self.window_size
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
    