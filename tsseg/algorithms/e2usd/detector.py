"""E2USD state detector: a wrapper around the reference implementation."""

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


class E2USDDetector(BaseSegmenter):
    """E2USD: unsupervised state detection from decomposed window embeddings.

    An encoder trained without labels on the series itself maps sliding
    windows to embeddings; a Dirichlet process Gaussian mixture (DPGMM)
    clusters the embeddings; each time point then takes the state that most
    windows covering it received [1]_.

    Parameters
    ----------
    axis : int
        Axis along which to segment if passed a multivariate series (2D input). If axis
        is 0, it is assumed each column is a time series and each row is a
        timepoint. i.e. the shape of the data is ``(n_timepoints,n_channels)``.
        ``axis == 1`` indicates the time series are in rows, i.e. the shape of the data
        is ``(n_channels, n_timepoints)``.
    window_size : int, default=256
        Length of the sliding windows, in time points. Must be even: an odd
        value is accepted by the constructor and by ``fit``, but ``predict``
        raises ``ValueError``.
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
    depth : int, default=1
        Passed to the encoder network, which does not use it: this value has
        no effect.
    out_channels : int, default=4
        Dimension of the window embeddings.
    reduced_size : int, default=80
        Number of output channels of each of the two convolutions of the
        encoder (one for the trend, one for the seasonal part), max-pooled
        over time before the linear layers.
    kernel_size : int, default=3
        Kernel size of the two convolutions of the encoder. At most
        ``min(window_size, 64)``, the length of a window after the low-pass
        filter of the encoder: a larger value makes ``fit`` raise
        ``RuntimeError``.
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
    variance computed over all channels together (a constant series becomes
    NaN, and ``predict`` raises ``ValueError``). ``fit`` then trains the
    encoder for ``nb_steps`` steps.

    The encoder (DDEM, decomposed dual-view embedding module [1]_) keeps the
    33 lowest frequencies of the Fourier transform of each window, which
    resamples a window of 64 points or more to 64 points, then splits it
    into a trend (moving average over 5 points) and a seasonal part (the
    residual). Each part goes through its own convolution, a max pooling over
    time and a linear layer; a last linear layer combines the two. The
    convolutions keep their random initial weights: only the linear layers
    are trained. At each step, the loss (FNCC, false negative cancellation
    contrastive learning [1]_) draws 20 random positions and, at each
    position, 4 windows shifted by one point. It pulls together the
    embeddings of the windows of a position. Among the pairs of windows from
    different positions, it pushes apart the half with the lowest product of
    their trend and seasonal similarities, and leaves out the other half as
    likely false negatives (windows of the same state). The positions are
    drawn among the first ``n_timepoints - 2 * window_size + 1`` points, so
    the last ``window_size - 3`` points never enter a training window.

    ``predict`` embeds the ``(n_timepoints - window_size) // step + 1``
    windows of the series, fits the DPGMM (full covariances, k-means
    initialisation) on these embeddings and numbers the window labels in
    order of first appearance. Each window votes for its label on the points
    it covers, and each point takes the label with the most votes (the
    smallest label on a tie). The last points of the series, fewer than
    ``step``, lie in no window: they take the label of the last window (the
    reference implementation gives them label 0).

    ``fit`` raises ``ValueError`` on a series of fewer than
    ``2 * window_size`` points; ``predict`` labels every point of such a
    series 0. ``y`` is ignored.

    The wrapped code is that of the reference implementation
    (https://github.com/AI4CTS/E2Usd), which publishes no licence. Its
    ``Dataset`` class comes from the code of Franceschi et al. (2019)
    (Apache License 2.0) and its ``moving_avg`` and ``series_decomp`` blocks
    from Autoformer (MIT licence); the ``LICENSE`` file of
    ``tsseg/algorithms/e2usd`` holds these two licences and covers these
    parts only.

    References
    ----------
    .. [1] Z. Lai, H. Li, D. Zhang, Y. Zhao, W. Qian and C. S. Jensen,
       "E2Usd: Efficient-yet-effective Unsupervised State Detection for
       Multivariate Time Series", Proceedings of the ACM Web Conference 2024
       (WWW '24), pp. 3010-3021, doi: 10.1145/3589334.3645593.
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
            description="Depth of the DDEM network.",
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
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="CNN kernel size.",
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
        depth=1,
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

    def _fit(self, X, y=None):
        """
        Train the E2USD encoder on X.

        Parameters
        ----------
        X : np.ndarray of shape (n_timepoints, n_channels)
            The training time series.
        y : np.ndarray, optional
            Ignored.

        Returns
        -------
        self : E2USDDetector
            The fitted detector.
        """
        # Deferred: torch is optional (checked at construction).
        import torch

        from .e2usd import DPGMM, E2USD_Adaper, params

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
        e2usd_params.update(
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
        np.ndarray of shape (n_timepoints,)
            Integer state labels, numbered from 0 in order of first
            appearance among the windows. All zeros if ``X`` has fewer than
            ``2 * window_size`` points.
        """
        from .e2usd import E2USD

        self._check_is_fitted()
        X = self._validate_data(X)
        n_timepoints, _ = X.shape

        # Handle short series case
        if (
            getattr(self, "_short_series_fitted", False)
            or n_timepoints < self.window_size * 2
        ):
            return np.zeros(n_timepoints, dtype=int)

        # Initialize the main model with fitted components
        e2usd_model = E2USD(
            win_size=self.window_size,
            step=self.step,
            encoder=self.encoder_,
            clustering_component=self.clustering_,
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
