"""Time2State state detector: a wrapper around the reference implementation."""

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    DependsOn,
    HasType,
    Interval,
    ParamDef,
)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # reported by the dependency check at construction
        return False
    return torch.cuda.is_available()


class Time2StateDetector(BaseSegmenter):
    """Time2State: unsupervised state detection from learned window embeddings.

    A causal CNN encoder, trained without labels on the series itself with the
    LSE loss, maps sliding windows to embeddings; a Dirichlet process Gaussian
    mixture (DPGMM) clusters the embeddings; each time point then takes the
    state that most windows covering it received [1]_.

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)``.
    window_size : int, default=256
        Length of the sliding windows, in time points. Must be even: ``fit``
        raises ``ValueError`` on an odd value.
    step : int, default=50
        Distance between the starts of consecutive windows, at most
        ``window_size``.
    n_states : int, default=20
        Upper bound on the number of states: the number of components of the
        DPGMM, capped at the number of windows. The DPGMM can leave
        components empty, so fewer states may be returned.
    alpha : float, default=1e3
        Weight concentration prior of the Dirichlet process
        (``weight_concentration_prior`` of
        ``sklearn.mixture.BayesianGaussianMixture``); larger values favour
        more states.
    batch_size : int, default=1
        Batch size of the training data loader. The training set is a single
        sample, the whole series, so this value has no effect.
    nb_steps : int, default=20
        Number of optimisation steps of the encoder.
    lr : float, default=0.003
        Learning rate of the Adam optimiser.
    depth : int, default=10
        Number of causal convolution blocks before the last one; the dilation
        doubles from one block to the next.
    out_channels : int, default=4
        Dimension of the window embeddings.
    reduced_size : int, default=80
        Number of channels output by the causal CNN, max-pooled over time
        before the final linear layer.
    kernel_size : int, default=3
        Kernel size of the convolutions of the causal CNN. Must be at least
        2: ``fit`` raises ``ValueError`` on 1.
    use_gpu : bool, optional
        Whether to run the encoder on CUDA device 0. ``None`` uses it when
        available; ``True`` without CUDA falls back to the CPU. The attribute
        stores the resolved boolean.
    random_state : int, optional
        Seed of PyTorch and of NumPy's global generator, set at the start of
        ``fit``. The DPGMM of ``predict`` draws from NumPy's global generator
        without its own seed: ``fit_predict`` with a seed is reproducible on
        one machine, but two ``predict`` calls after one ``fit`` can differ.

    Notes
    -----
    ``fit`` and ``predict`` z-normalise the series with one mean and one
    variance computed over all channels together; a series of zero variance
    is only centred (the reference implementation returns NaN), so a constant
    series is labelled as one state. ``fit`` then trains the
    encoder for ``nb_steps`` steps. At each step, the LSE loss draws 10 random positions and, at each
    position, 4 windows shifted by one point; it pulls the embeddings of the
    windows of a position together and pushes the mean embeddings of
    different positions apart [1]_.

    ``predict`` embeds the ``(n_timepoints - window_size) // step + 1``
    windows of the series, fits the DPGMM (full covariances, k-means
    initialisation) on these embeddings and numbers the window labels in
    order of first appearance. Each window votes for its label on the points
    it covers, and each point takes the label with the most votes (the
    smallest label on a tie). The last points of the series, fewer than
    ``step``, lie in no window: they take the label of the last window (the
    reference implementation gives them label 0).

    ``fit`` raises ``ValueError`` on a series of fewer than
    ``2 * window_size`` points. A series of at most ``2 * window_size``
    points is not encoded: if it is the one given to ``fit``, or the one
    given to ``predict``, ``predict`` labels every point 0.

    The wrapped code is that of the reference implementation
    (https://github.com/Lab-ANT/Time2State, MIT licence), whose causal CNN
    encoder comes from the code of Franceschi et al. (2019) (Apache License
    2.0, partly from locuslab/TCN, MIT licence); the ``LICENSE`` file of
    ``tsseg/algorithms/time2state`` holds the three licences.

    References
    ----------
    .. [1] C. Wang, K. Wu, T. Zhou and Z. Cai, "Time2State: An Unsupervised
       Framework for Inferring the Latent States in Time Series Data",
       Proceedings of the ACM on Management of Data 1(1), 2023 (SIGMOD),
       doi: 10.1145/3588697.
    """

    _tags = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "detector_type": "state_detection",
        "python_dependencies": ["torch"],
        "non_deterministic": True,
        "cant_pickle": True,
        "capability:unsupervised": True,  # n_states is an upper bound
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "window_size": ParamDef(
            constraint=Interval(int, 4, None, Closed.LEFT),
            description="Sliding window size.",
            group="windowing",
        ),
        "step": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Step size of the sliding window.",
            group="windowing",
        ),
        "n_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Maximum number of states for DPGMM.",
        ),
        "alpha": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Concentration parameter for DPGMM.",
        ),
        "batch_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Training batch size.",
            group="training",
        ),
        "nb_steps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of optimisation steps.",
            group="training",
        ),
        "lr": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Learning rate.",
            group="training",
        ),
        "depth": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Depth of the Causal CNN.",
            group="architecture",
        ),
        "out_channels": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Encoder output channels.",
            group="architecture",
        ),
        "reduced_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Size of CNN output before final linear layer.",
            group="architecture",
        ),
        "kernel_size": ParamDef(
            # 1 would make each causal block chomp its whole output (padding 0).
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="CNN kernel size (at least 2).",
            group="architecture",
        ),
        "random_state": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Random seed.",
            nullable=True,
        ),
        "use_gpu": ParamDef(
            constraint=HasType((bool,)),
            description="Use GPU if available (None = auto-detect).",
            nullable=True,
            ui_hidden=True,
        ),
        "_cross_constraints": [
            DependsOn(
                "step <= window_size",
                "step must be <= window_size",
            ),
            DependsOn(
                "window_size % 2 == 0",
                "window_size must be even",
            ),
            DataDependent(
                "window_size * 2 <= n_samples",
                "Series must have at least 2 * window_size samples",
            ),
        ],
    }

    def __init__(
        self,
        axis=0,
        window_size=256,
        step=50,
        n_states=20,
        alpha=1e3,
        batch_size=1,
        nb_steps=20,
        lr=0.003,
        depth=10,
        out_channels=4,
        reduced_size=80,
        kernel_size=3,
        use_gpu=None,
        random_state=None,
    ):
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
            self.use_gpu = _cuda_available()
        else:
            self.use_gpu = use_gpu and _cuda_available()

        super().__init__(axis=axis)

    def _fit(self, X, y=None, axis=None):
        """Train the encoder on ``X``.

        Only the encoder is trained here; the DPGMM is fitted in ``predict``,
        on the embeddings of the series to segment.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels) or (n_channels, n_timepoints)
            The training time series. Shape depends on ``self.axis``.
        y : None
            Not used, for API compatibility.
        axis : None
            Not used.

        Returns
        -------
        self : Time2StateDetector
            The fitted detector. If ``n_timepoints <= 2 * window_size``, no
            encoder is trained and ``predict`` labels every point 0.
        """
        # Deferred: torch is optional (checked at construction).
        import torch

        from .time2state import DPGMM, CausalConv_LSE_Adaper, params_LSE

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
        t2s_params.update(
            {
                "win_size": self.window_size,
                "batch_size": self.batch_size,
                "nb_steps": self.nb_steps,
                "lr": self.lr,
                "depth": self.depth,
                "out_channels": self.out_channels,
                "reduced_size": self.reduced_size,
                "kernel_size": self.kernel_size,
                "cuda": self.use_gpu,
                "gpu": 0,  # Assuming GPU 0 if used
                "in_channels": n_channels,
            }
        )
        self.encoder_ = CausalConv_LSE_Adaper(t2s_params)

        # 2. Initialize Clustering component
        self.clustering_ = DPGMM(n_states=self.n_states, alpha=self.alpha)

        # 3. Fit the encoder
        self.encoder_.fit(X)

        return self

    def _predict(self, X, axis=None):
        """Label each time point of ``X`` with a state.

        Embeds the sliding windows of ``X`` with the trained encoder, fits the
        DPGMM on these embeddings, then gives each time point the state that
        most of the windows covering it received.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels) or (n_channels, n_timepoints)
            The time series to segment. Shape depends on ``self.axis``.
        axis : None
            Not used.

        Returns
        -------
        np.ndarray of shape (n_timepoints,)
            Integer state labels, numbered from 0 in order of first
            appearance among the windows. All zeros if ``X`` or the fitted
            series has at most ``2 * window_size`` points.
        """
        from .time2state import Time2State

        self._check_is_fitted()
        X = self._validate_data(X)
        n_timepoints, n_channels = X.shape

        # Handle short series case from fit or if predict X is short
        if (
            getattr(self, "_short_series_fitted", False)
            or n_timepoints <= self.window_size * 2
        ):
            if self.get_tag("returns_dense"):
                return np.array([], dtype=int)  # No change points for short series
            else:
                return np.zeros(n_timepoints, dtype=int)  # All points in segment 0

        # Initialize the main model with fitted components
        t2s_model = Time2State(
            win_size=self.window_size,
            step=self.step,
            encoder=self.encoder_,
            clustering_component=self.clustering_,
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
                if state_sequence[i] != state_sequence[i - 1]:
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
