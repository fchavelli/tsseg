"""
This module provides a wrapper for the PaTSS algorithm to integrate it
with the tsseg library's API, following the aeon BaseSegmenter structure.
"""
import numpy as np
import pandas as pd
import tempfile
from ..base import BaseSegmenter

# Import the core logic from the old, working implementation
from .algorithms.PaTSS_perso import run_patss

def _transform_to_dfs(time_series: np.ndarray) -> list:
    """
    Transforms a NumPy array into a list of pandas DataFrames as expected by run_patss.
    """
    if time_series.ndim == 1:
        time_series = time_series.reshape(-1, 1)
    
    dfs = []
    for i in range(time_series.shape[1]):
        df = pd.DataFrame({
            'average_value': time_series[:, i],
            'time': np.arange(time_series.shape[0])
        })
        dfs.append(df)
    
    return dfs

class PatssDetector(BaseSegmenter):
    """
    A wrapper for the PaTSS (Pattern-based Time Series Segmentation) algorithm,
    compatible with the aeon framework.

    This implementation uses the original logic from PaTSS_perso.py and adapts it
    to the BaseSegmenter API.

    Parameters
    ----------
    config : dict, optional
        A dictionary containing the settings to use within PaTSS. If not provided,
        the default hyperparameters defined in PaTSS_perso.py will be used.
    axis : int, default=0
        The axis of the time series data. Assumes (n_timepoints, n_channels).
    """
    _tags = {
        "X_inner_type": "np.ndarray",
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "detector_type": "state_detection",
        "fit_is_empty": True,  # PaTSS is unsupervised and has no separate fit stage
        "returns_dense": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    def __init__(self, config: dict = None, axis: int = 0):
        self.config = config
        # The original PaTSS implementation expects (n_timepoints, n_channels),
        # which corresponds to axis 0 in aeon's convention.
        super().__init__(axis=axis)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Segments the time series using the PaTSS algorithm.

        Parameters
        ----------
        X : np.ndarray
            The time series to segment, with shape (n_timepoints, n_channels).

        Returns
        -------
        np.ndarray
            A 1D array of integer state labels for each time point.
        """
        # 1. Convert the input NumPy array to the list of DataFrames format
        #    expected by the original run_patss function.
        multivariate_time_series = _transform_to_dfs(X)

        # 2. PaTSS requires a directory for temporary files. We use a context
        #    manager to handle this cleanly.
        with tempfile.TemporaryDirectory() as temp_dir:
            patss_output = run_patss(
                directory=temp_dir,
                multivariate_time_series=multivariate_time_series,
                length_time_series=X.shape[0],
                config=self.config
            )

        # 3. Handle the case where no segmentation is returned.
        #    This happens if no patterns are found or if run_patss returns None.
        if patss_output is None:
            # If run_patss returns None (which shouldn't happen normally), treat it as failure
            segmentation_probas = None
        else:
            segmentation_probas, _, _ = patss_output

        if not isinstance(segmentation_probas, np.ndarray) or segmentation_probas.size == 0:
            # Return a single segment (all zeros)
            return np.zeros(X.shape[0], dtype=int)

        # 4. Convert the probability matrix to a dense array of state labels.
        #    The original implementation returns probabilities, so we take the argmax.
        labels = np.argmax(segmentation_probas, axis=0)
        
        return labels

    def _fit(self, X, y=None):
        """
        The _fit method is empty for PaTSS as it is an unsupervised algorithm
        that performs all its work in the _predict step.
        """
        # This method is intentionally left empty because fit_is_empty is True.
        return self
