"""
This module provides an aeon-compatible wrapper for the TICC algorithm.
"""
import numpy as np
from ..base import BaseSegmenter
from .ticc import TICC

class TiccDetector(BaseSegmenter):
    """
    An aeon-compatible wrapper for the TICC (Toeplitz Inverse Covariance-based
    Clustering) algorithm for time series segmentation.

    Parameters
    ----------
    window_size : int, default=10
        The size of the sliding window.
    n_states : int, default=5
        The number of states (clusters) to find.
    lambda_parameter : float, default=11e-2
        Sparsity parameter.
    beta : float, default=400
        Switching penalty, controls the temporal consistency.
    maxIters : int, default=100
        Maximum number of iterations for the TICC solver.
    threshold : float, default=2e-5
        Convergence threshold.
    num_proc : int, default=1
        Number of processes to use for parallel computation.
    axis : int, default=0
        The axis of the input series to segment.
    """
    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    def __init__(self, window_size=10, n_states=5,
                 lambda_parameter=11e-2, beta=400, maxIters=100,
                 threshold=2e-5, num_proc=1, axis=0):
        self.window_size = window_size
        self.n_states = n_states
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.num_proc = num_proc
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        """
        Fit the TICC model to the data.
        """

        # The TICC implementation expects (n_samples, n_features)
        # where n_samples is the number of timepoints.
        self.ticc_ = TICC(
            window_size=self.window_size,
            number_of_clusters=self.n_states,
            lambda_parameter=self.lambda_parameter,
            beta=self.beta,
            maxIters=self.maxIters,
            threshold=self.threshold,
            num_proc=self.num_proc
        )

        # The fit_transform method trains the model and returns the segmentation
        clustered_points, _ = self.ticc_.fit_transform(X)
        
        # The TICC algorithm labels each window, resulting in fewer labels than timepoints.
        # We pad the labels to match the original time series length by repeating the last label.
        n_timepoints = X.shape[0]
        self._state_labels = np.zeros(n_timepoints, dtype=int)
        self._state_labels[:len(clustered_points)] = clustered_points
        self._state_labels[len(clustered_points):] = clustered_points[-1]

        return self

    def _predict(self, X):
        """
        Return the segmentation labels found during fit.

        Parameters
        ----------
        X : np.ndarray
            The time series to segment of shape (n_channels, n_timepoints).
        
        Returns
        -------
        np.ndarray
            The predicted state labels as a 1D array of shape (n_timepoints,).
        """
        self._check_is_fitted()
        # TICC's fit_transform already computes the segmentation for the training data.
        # For simplicity, this wrapper is a "fit-and-predict" type.
        # A true predict would require re-using the learned inverse covariance matrices.
        return self._state_labels
