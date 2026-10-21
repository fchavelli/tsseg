"""
This module provides an aeon-compatible wrapper for the TICC algorithm.
"""

import warnings

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    Interval,
    ParamDef,
)
from .ticc import TICC, n_free_parameters


class TiccDetector(BaseSegmenter):
    """
    An aeon-compatible wrapper for the TICC (Toeplitz Inverse Covariance-based
    Clustering) algorithm for time series segmentation.

    Parameters
    ----------
    window_size : int, default=10
        The size of the sliding window.
    n_states : int or None, default=None
        The number of states (clusters) to find; ``1`` labels every time
        point with the same state. ``None`` selects it by the Bayesian
        information criterion among ``1, ..., max_states`` (see Notes).
    max_states : int, default=8
        Largest number of states tried when ``n_states`` is ``None``.
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

    Attributes
    ----------
    n_states_ : int
        Number of states of the fitted model.
    bic_ : dict
        BIC of each candidate number of states (only when ``n_states=None``);
        ``inf`` for a number of states the solver failed to fit.

    Notes
    -----
    The paper selects the number of clusters :math:`K` by BIC [1]_ (Sec. 3,
    "Selecting the Number of Clusters", and Sec. 7, where BIC picks
    :math:`K = 5`). With ``n_states=None``, TICC is fitted for each
    :math:`K` in ``1, ..., max_states`` and the fit with the lowest

    .. math::

        \\mathrm{BIC}(K) = -\\frac{2}{w} \\sum_t \\ell\\ell(X_t, \\Theta_{c(t)})
        + \\Big(\\sum_{i=1}^{K} (p_i + d) + s\\Big) \\log T

    is kept, where :math:`\\ell\\ell` is the Gaussian log-likelihood of the
    stacked window :math:`X_t` under the precision :math:`\\Theta_{c(t)}` of
    its cluster (Eq. 2), :math:`p_i` the number of free entries of the block
    Toeplitz :math:`\\Theta_i` above ``2e-5`` in absolute value, :math:`d` the
    number of channels (the mean), :math:`s` the number of switches between
    clusters and :math:`T` the number of time points. A time point belongs to
    :math:`w` = ``window_size`` overlapping windows, so the sum of the window
    log-likelihoods counts each observation :math:`w` times: it is divided by
    :math:`w` (a composite likelihood). This costs ``max_states`` fits of
    TICC.

    On univariate series, the log-likelihood of the fitted model often does
    not increase with the number of states, and BIC then selects a single
    state.

    References
    ----------
    .. [1] D. Hallac, S. Vare, S. Boyd and J. Leskovec, "Toeplitz Inverse
       Covariance-Based Clustering of Multivariate Time Series Data", KDD 2017,
       doi: 10.1145/3097983.3098060.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "window_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Sliding window size.",
        ),
        "n_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of clusters/states to identify (None: chosen by BIC).",
            nullable=True,
        ),
        "max_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Largest number of states tried by BIC.",
        ),
        "lambda_parameter": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Sparsity regularisation parameter.",
        ),
        "beta": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Switching penalty for temporal consistency.",
        ),
        "maxIters": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Maximum EM iterations.",
        ),
        "threshold": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Convergence threshold.",
        ),
        "num_proc": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of parallel processes.",
        ),
        "_cross_constraints": [
            DataDependent(
                "window_size < n_samples",
                "window_size must be less than the series length",
            ),
        ],
    }

    def __init__(
        self,
        window_size=10,
        n_states=None,
        lambda_parameter=11e-2,
        beta=400,
        maxIters=100,
        threshold=2e-5,
        num_proc=1,
        max_states=8,
        axis=0,
    ):
        self.window_size = window_size
        self.n_states = n_states
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.maxIters = maxIters
        self.threshold = threshold
        self.num_proc = num_proc
        self.max_states = max_states
        super().__init__(axis=axis)

    def _fit_ticc(self, X, n_states):
        """Fit TICC with ``n_states`` clusters; return the solver and its labels."""
        model = TICC(
            window_size=self.window_size,
            number_of_clusters=n_states,
            lambda_parameter=self.lambda_parameter,
            beta=self.beta,
            maxIters=self.maxIters,
            threshold=self.threshold,
            num_proc=self.num_proc,
        )
        clustered_points, _ = model.fit_transform(X)
        return model, np.asarray(clustered_points, dtype=int)

    def _bic(self, model, labels, n_timepoints, n_channels):
        """BIC of a fitted TICC model (see the Notes of the class)."""
        valid = labels[: len(labels) - (self.window_size - 1)]
        n_params = sum(
            n_free_parameters(theta, self.window_size, n_channels) + n_channels
            for cluster, theta in model.train_cluster_inverse_.items()
            if np.any(valid == cluster)
        )
        n_switches = int(np.count_nonzero(np.diff(labels)))
        loglik = model.log_likelihood()
        return -2.0 * loglik / self.window_size + (n_params + n_switches) * np.log(
            n_timepoints
        )

    def _fit(self, X, y=None):
        """
        Fit the TICC model to the data.

        Parameters
        ----------
        X : np.ndarray
            The time series, of shape (n_timepoints, n_channels).
        y : None
            Not used.

        Returns
        -------
        self
        """
        n_timepoints, n_channels = X.shape
        if self.n_states is not None:
            self.ticc_, clustered_points = self._fit_ticc(X, self.n_states)
            self.n_states_ = self.n_states
        else:
            self.bic_ = {}
            best = None
            for n_states in range(1, self.max_states + 1):
                try:
                    model, labels = self._fit_ticc(X, n_states)
                    score = self._bic(model, labels, n_timepoints, n_channels)
                except (np.linalg.LinAlgError, KeyError, ValueError):
                    self.bic_[n_states] = np.inf
                    continue
                self.bic_[n_states] = float(score)
                if best is None or score < best[0]:
                    best = (score, n_states, model, labels)
            if best is None:
                raise RuntimeError(
                    f"{type(self).__name__}: TICC failed for every number of "
                    f"states from 1 to {self.max_states}."
                )
            _, self.n_states_, self.ticc_, clustered_points = best
            if self.n_states_ == self.max_states and self.max_states > 1:
                warnings.warn(
                    f"{type(self).__name__}: BIC selected the largest number of "
                    f"states tried (max_states={self.max_states}); a larger "
                    "max_states may fit better.",
                    UserWarning,
                    stacklevel=3,
                )

        # The TICC algorithm labels each window, resulting in fewer labels than timepoints.
        # We pad the labels to match the original time series length by repeating the last label.
        self._state_labels = np.zeros(n_timepoints, dtype=int)
        self._state_labels[: len(clustered_points)] = clustered_points
        self._state_labels[len(clustered_points) :] = clustered_points[-1]

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
