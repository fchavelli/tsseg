"""
This module provides an aeon-compatible wrapper for the E2USD algorithm.
"""
import numpy as np
import torch
from ..base import BaseSegmenter
from .e2usd import E2USD, E2USD_Adaper, DPGMM, params

class E2USDDetector(BaseSegmenter):
    """
    A wrapper for the E2USD algorithm, compatible with the aeon library.

    E2USD uses a deep learning based encoder to learn representations of
    time series windows, which are then clustered to identify states.

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input).
        Default is 0, assuming shape (n_timepoints, n_channels).
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
    depth : int, default=1
        Depth of the DDEM Network.
    out_channels : int, default=4
        Number of output channels of the encoder.
    reduced_size : int, default=80
        Size of the output of the CNN before the final linear layer.
    kernel_size : int, default=3
        Kernel size for the convolutions in the CNN.
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
                 batch_size=1, nb_steps=20, lr=0.003, depth=1,
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

    def _fit(self, X, y=None):
        """
        Fit the E2-USD model.

        This involves training the DDEM encoder on the input time series X.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The training time series.
        y : np.ndarray, optional
            The target segmentation. If provided, n_states is inferred from y.
        """


        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = self._validate_data(X)
        n_timepoints, n_channels = X.shape

        # Handle cases where the series is too short for windowing
        if n_timepoints < self.window_size * 2:
            self._short_series_fitted = True
            return self

        self._short_series_fitted = False

        # 1. Initialize Encoder
        e2usd_params = params.copy()
        e2usd_params.update({
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
        self.encoder_ = E2USD_Adaper(e2usd_params)

        # 2. Initialize Clustering component
        self.clustering_ = DPGMM(n_states=self.n_states, alpha=self.alpha)

        # 3. Fit the encoder
        self.encoder_.fit(X)

        return self

    def _predict(self, X):
        """
        Predict the state sequence for the input time series X.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The time series to segment.
        
        Returns
        -------
        np.ndarray
            An array of segment labels for each time point.
        """
        self._check_is_fitted()
        X = self._validate_data(X)
        n_timepoints, _ = X.shape

        # Handle short series case
        if getattr(self, '_short_series_fitted', False) or n_timepoints < self.window_size * 2:
            return np.zeros(n_timepoints, dtype=int)

        # Initialize the main model with fitted components
        e2usd_model = E2USD(
            win_size=self.window_size,
            step=self.step,
            encoder=self.encoder_,
            clustering_component=self.clustering_
        )

        # Use the internal methods to predict without re-fitting the encoder
        e2usd_model._E2USD__length = X.shape[0]
        e2usd_model._E2USD__encode(X, self.window_size, self.step)
        e2usd_model._E2USD__cluster()
        e2usd_model._E2USD__assign_label()

        return e2usd_model.state_seq
    
    def _validate_data(self, X):
        """Validate and reshape input data according to axis."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim == 2:
            if self.axis == 1:
                X = X.T
        else:
            raise ValueError(f"Input X must be 1D or 2D, got {X.ndim}D")
        
        return X