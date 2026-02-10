"""
This module provides an aeon-compatible wrapper for the Time2State algorithm.
"""
import numpy as np
import torch
from ..base import BaseSegmenter
from .time2state import Time2State, CausalConv_LSE_Adaper, DPGMM, params_LSE

class Time2StateDetector(BaseSegmenter):
    """
    A wrapper for the Time2State algorithm for time series segmentation,
    compatible with the aeon library.

    Time2State uses a Causal CNN based encoder to learn representations of
    time series windows, which are then clustered to identify states.

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)``.
    window_size : int, default=256
        The size of the sliding window.
    step : int, default=50
        The step size of the sliding window.
    n_states : int, default=20
        The maximum number of states for the DPGMM clustering.
    alpha : float, default=1e3
        The concentration parameter for the DPGMM clustering.
    batch_size : int, default=1
        Batch size for training the neural network.
    nb_steps : int, default=20
        Number of optimization steps for training.
    lr : float, default=0.003
        Learning rate for the optimizer.
    depth : int, default=10
        Depth of the Causal CNN.
    out_channels : int, default=4
        Number of output channels of the encoder.
    reduced_size : int, default=80
        Size of the output of the Causal CNN before the final linear layer.
    kernel_size : int, default=3
        Kernel size for the convolutions in the Causal CNN.
    use_gpu : bool, optional
        Whether to use GPU if available. If None, it will be auto-detected.
    random_state : int, optional
        Random state for reproducibility.
    """
    _tags = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "detector_type": "state_detection",
        "capability:unsupervised": True,        # n_states is an upper bound
        "capability:semi_supervised": True,
    }

    def __init__(self, axis=0, window_size=256, step=50, n_states=20, alpha=1e3,
                 batch_size=1, nb_steps=20, lr=0.003, depth=10,
                 out_channels=4, reduced_size=80, kernel_size=3,
                 use_gpu=None, random_state=None):

        self.window_size = window_size
        self.step = step
        self.n_states = n_states
        self.alpha = alpha
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.depth = depth
        self.out_channels = out_channels
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size
        self.random_state = random_state
        
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()

        super().__init__(axis=axis)

    def _fit(self, X, y=None, axis=None):
        """
        Fit the Time2State model.

        This involves training the Causal CNN encoder on the input time series X.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels) or (n_channels, n_timepoints)
            The training time series. Shape depends on self.axis.
        y : None
            Not used, for API compatibility.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = self._validate_data(X)
        n_timepoints, n_channels = X.shape

        # Handle cases where the series is too short for windowing
        if n_timepoints <= self.window_size * 2:
            # Set a flag to indicate the model is fitted for short series
            self._short_series_fitted = True
            self._is_fitted = True
            return self

        self._short_series_fitted = False

        # 1. Initialize Encoder
        t2s_params = params_LSE.copy()
        t2s_params.update({
            "win_size": self.window_size,
            "batch_size": self.batch_size,
            "nb_steps": self.nb_steps,
            "lr": self.lr,
            "depth": self.depth,
            "out_channels": self.out_channels,
            "reduced_size": self.reduced_size,
            "kernel_size": self.kernel_size,
            "cuda": self.use_gpu,
            "gpu": 0, # Assuming GPU 0 if used
            "in_channels": n_channels,
        })
        self.encoder_ = CausalConv_LSE_Adaper(t2s_params)

        # 2. Initialize Clustering component
        self.clustering_ = DPGMM(n_states=self.n_states, alpha=self.alpha)

        # 3. Fit the encoder
        self.encoder_.fit(X)

        return self

    def _predict(self, X, axis=None):
        """
        Predict the state sequence for the input time series X.

        This uses the already-fitted encoder and clustering model.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels) or (n_channels, n_timepoints)
            The time series to segment. Shape depends on self.axis.
        
        Returns
        -------
        np.ndarray
            If returns_dense is True (default): array of change point locations.
            If returns_dense is False: array of segment labels for each time point.
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        n_timepoints, n_channels = X.shape
        

        # Handle short series case from fit or if predict X is short
        if getattr(self, '_short_series_fitted', False) or n_timepoints <= self.window_size * 2:
            if self.get_tag("returns_dense"):
                return np.array([], dtype=int)  # No change points for short series
            else:
                return np.zeros(n_timepoints, dtype=int)  # All points in segment 0

        # Initialize the main model with fitted components
        t2s_model = Time2State(
            win_size=self.window_size,
            step=self.step,
            encoder=self.encoder_,
            clustering_component=self.clustering_
        )

        # Use the internal methods to predict without re-fitting the encoder
        t2s_model._Time2State__length = X.shape[0]
        t2s_model._Time2State__encode(X, self.window_size, self.step)
        t2s_model._Time2State__cluster()
        t2s_model._Time2State__assign_label()

        state_sequence = t2s_model.state_seq
        
        # Convert to change points (dense representation)
        if self.get_tag("returns_dense"):
            change_points = []
            for i in range(1, len(state_sequence)):
                if state_sequence[i] != state_sequence[i-1]:
                    change_points.append(i)
            return np.array(change_points, dtype=int)
        else:
            return state_sequence
    
    def _validate_data(self, X):
        """Validate and reshape input data according to axis."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 2:
            # Handle axis orientation
            if self.axis == 1:
                # Time series are in rows, transpose to (n_timepoints, n_channels)
                X = X.T
        else:
            raise ValueError(f"Input X must be 1D or 2D, got {X.ndim}D")
        
        return X
