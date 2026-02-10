import numpy as np
from ..base import BaseSegmenter

# Import pyhsmm after tsseg to ensure compatibility patches are applied
try:
    from pyhsmm import distributions, models
    from pyhsmm.util.text import progprint_xrange
    _HAS_PYHSMM = True
except ImportError:
    _HAS_PYHSMM = False


class HdpHsmmLegacyDetector(BaseSegmenter):
    """
    HDP-HSMM segmentation using the pyhsmm library.

    This class wraps the WeakLimitHDPHSMM model from the pyhsmm library to
    provide a state-based segmentation of a time series.

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)``.
    alpha : float, default=1.0
        Concentration parameter for the dirichlet process prior on durations.
    beta : float, default=1.0
        Rate parameter for the dirichlet process prior on durations.
    n_iter : int, default=20
        Number of Gibbs sampling iterations for the model to run.
    n_max_states : int, default=60
        Weak limit truncation level (maximum number of states).
    """

    _tags = {
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
    }

    def __init__(self, axis=0, alpha=1.0, beta=1.0, n_iter=20, n_max_states=60):
        if not _HAS_PYHSMM:
            raise ImportError("pyhsmm is not installed. Please install it to use HdpHsmmLegacyDetector.")
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.n_max_states = n_max_states
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        """Fit the HDP-HSMM model to the data.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels) or (n_channels, n_timepoints)
            The time series to segment. Shape depends on self.axis.
        y : None
            Ignored. For API compatibility.
        """
        X = self._validate_data(X)
        obs_dim = X.shape[1]
        obs_hypparams = {
            "mu_0": np.zeros(obs_dim),
            "sigma_0": np.eye(obs_dim),
            "kappa_0": 0.25,
            "nu_0": obs_dim + 2,
        }
        dur_hypparams = {"alpha_0": self.alpha, "beta_0": self.beta}

        obs_distns = [
            distributions.Gaussian(**obs_hypparams) for _ in range(self.n_max_states)
        ]
        dur_distns = [
            distributions.PoissonDuration(**dur_hypparams) for _ in range(self.n_max_states)
        ]

        self.model_ = models.WeakLimitHDPHSMM(
            alpha=6.0,
            gamma=6.0,
            init_state_concentration=6.0,
            obs_distns=obs_distns,
            dur_distns=dur_distns,
        )
        # Duration truncation speeds things up
        self.model_.add_data(X, trunc=600)

        for _ in progprint_xrange(self.n_iter):
            self.model_.resample_model()

        return self

    def _predict(self, X):
        """Return the change points predicted by the fitted model.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels) or (n_channels, n_timepoints)
            The time series to segment. Shape depends on self.axis.

        Returns
        -------
        np.ndarray
            Array of segment labels for each time point.
        """
        X = self._validate_data(X)
        state_sequence = self.model_.stateseqs[0]

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
