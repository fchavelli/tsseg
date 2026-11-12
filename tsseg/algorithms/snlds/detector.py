"""SNLDS (Switching Non-Linear Dynamical System) detector for time series segmentation.

Wraps the CAVI-SNLDS model from:
    Dong, Z., Seybold, B., Murphy, K., & Bui, H. (2020).
    Collapsed Amortized Variational Inference for Switching Nonlinear Dynamical Systems.
    ICML 2020.

The model learns discrete regime labels via variational inference with a
forward-backward algorithm over discrete states, making it a natural state
detector.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import Closed, HasType, Interval, ParamDef, StrOptions

logger = logging.getLogger(__name__)


class SNLDSDetector(BaseSegmenter):
    """State detection by collapsed variational switching nonlinear dynamics.

    The observation encoder is bidirectional. The discrete regimes are
    marginalized exactly during training, conditional on continuous states.

    .. warning::
        Experimental. The validation documents unsuccessful seeds and series
        as well as gains. Current defaults are adaptations, not a reproduction
        of the original paper's experiments.

    Parameters
    ----------
    n_states : int, default=3
        Number of regimes. Reference labels are never used for fitting or
        initialization, even when supplied as ``y``.
    hidden_dim : int, default=4
        Continuous latent dimension.
    rnn_dim : int, default=16
        Units per encoder direction and in the optional autoregressive RNN.
    rnn_type : {"gru", "lstm", "simplernn"}, default="gru"
        Recurrent cell type.
    n_train_steps : int, default=1000
        Variational updates, in addition to initialization.
    learning_rate : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=8
        Random windows per update.
    train_window : int or None, default=64
        Window length after preprocessing and downsampling. None uses the
        full sequence, repeated ``batch_size`` times.
    objective : {"elbo", "iwae"}, default="elbo"
        Collapsed objective, using one latent draw per training window.
    inference_type : {"factorized", "autoregressive"}, default="factorized"
        Independent latent noise conditional on bidirectional context, or the
        paper's forward RNN additionally conditioned on past latent draws.
    init_steps : int, default=200
        Reconstruction updates followed by k-means on latent means. Cluster
        centers and predicted transitions initialize the dynamics. Zero
        disables initialization. Position-based clustering can be unsuitable
        for modes distinguished by velocity rather than position.
    decoding : {"viterbi", "marginal"}, default="viterbi"
        Conditional joint MAP regime path or pointwise posterior modes.
    posterior_mean : bool, default=True
        Decode at continuous posterior means (propagated conditional means
        for autoregressive inference), rather than one posterior draw. This
        plug-in approximation does not integrate continuous uncertainty.
    feature_window : int, default=1
        One uses raw data. Larger values replace channels by their centered
        local mean and standard deviation, with nearest padding, before
        normalization. This representation is an optional tsseg adaptation.
    downsample : int, default=1
        Observation sampling interval after feature extraction. Labels are
        expanded to the original length, including the final points.
    use_temperature_annealing : bool, default=True
        Anneal transition temperature down to one.
    temperature_init : float, default=10.0
        Initial transition temperature.
    use_cross_entropy_reg : bool, default=False
        Enable the original pointwise uniform-state regularizer. The exact
        collapsed objective does not require it (Ansari et al., Appendix B.4.1).
    xent_init : float, default=1000.0
        Initial regularizer weight when enabled.
    random_state : int or None, default=42
        Seed of weights, windows, clustering and stateless latent noise.
        TensorFlow's global seed is set during initialization.
    verbose : int, default=0
        Log the objective ten times during variational training when positive.
    axis : int, default=0
        Time axis of the input.

    Notes
    -----
    Gaussian scales use positive softplus parameters. Initial means and
    normalized regime logits are tracked by Keras. Stateless noise folds in
    update and time indices. Prediction reuses one graph across input lengths.
    Bitwise equivalence across devices or TensorFlow versions is not promised.

    Factorized inference, reconstruction initialization, Viterbi and local
    moments are adaptations of SNLDS. The autoregressive posterior remains
    available. None of these choices guarantees semantic activity segmentation
    or that every requested state will be used.

    References
    ----------
    .. [1] Dong et al. (2020). Collapsed Amortized Variational Inference for
       Switching Nonlinear Dynamical Systems. ICML, PMLR 119:2638-2647.
    .. [2] Ansari et al. (2021). Deep Explicit Duration Switching Models for
       Time Series. NeurIPS, Appendix B.4.1.
    """

    _tags: dict[str, Any] = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        # The model requires a fixed K and does not perform model selection.
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
        "python_dependencies": ["tensorflow", "tensorflow_probability", "tf_keras"],
        # Conservative training tag: floating-point results may differ across
        # devices even with stateless latent sampling.
        "non_deterministic": True,
    }

    _parameter_schema = {
        "n_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of discrete regimes.",
        ),
        "hidden_dim": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Dimension of the continuous latent state.",
            group="architecture",
        ),
        "rnn_dim": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Hidden size of the inference RNN.",
            group="architecture",
        ),
        "rnn_type": ParamDef(
            constraint=StrOptions({"gru", "lstm", "simplernn"}),
            description="Inference RNN cell.",
            group="architecture",
        ),
        "n_train_steps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of gradient steps.",
            group="training",
        ),
        "learning_rate": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Adam learning rate.",
            group="training",
        ),
        "batch_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Training windows per gradient step.",
            group="training",
        ),
        "train_window": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            nullable=True,
            description="Length of the training windows (None: whole series).",
            group="training",
        ),
        "objective": ParamDef(
            constraint=StrOptions({"elbo", "iwae"}),
            description="Training objective.",
            group="training",
        ),
        "inference_type": ParamDef(
            constraint=StrOptions({"autoregressive", "factorized"}),
            description="Variational posterior family after bidirectional encoding.",
            group="architecture",
        ),
        "init_steps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Reconstruction updates before clustering latent means.",
            group="training",
        ),
        "decoding": ParamDef(
            constraint=StrOptions({"viterbi", "marginal"}),
            description="Joint MAP regime path or pointwise posterior modes.",
        ),
        "posterior_mean": ParamDef(
            constraint=HasType((bool,)),
            description="Decode at posterior means instead of a sampled trajectory.",
        ),
        "feature_window": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Local mean/std window (1: raw observations).",
        ),
        "downsample": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Observation sampling interval; output keeps its original length.",
        ),
        "temperature_init": ParamDef(
            constraint=Interval(float, 1, None, Closed.LEFT),
            description="Initial temperature of the discrete transitions.",
            group="training",
        ),
        "xent_init": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Initial weight of the cross-entropy regulariser.",
            group="training",
        ),
    }

    def __init__(
        self,
        *,
        n_states: int = 3,
        hidden_dim: int = 4,
        rnn_dim: int = 16,
        rnn_type: str = "gru",
        n_train_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        train_window: int | None = 64,
        objective: str = "elbo",
        inference_type: str = "factorized",
        init_steps: int = 200,
        decoding: str = "viterbi",
        posterior_mean: bool = True,
        feature_window: int = 1,
        downsample: int = 1,
        use_temperature_annealing: bool = True,
        temperature_init: float = 10.0,
        use_cross_entropy_reg: bool = False,
        xent_init: float = 1000.0,
        random_state: int | None = 42,
        verbose: int = 0,
        axis: int = 0,
    ):
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        self.rnn_type = rnn_type
        self.n_train_steps = n_train_steps
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_window = train_window
        self.objective = objective
        self.inference_type = inference_type
        self.init_steps = init_steps
        self.decoding = decoding
        self.posterior_mean = posterior_mean
        self.feature_window = feature_window
        self.downsample = downsample
        self.use_temperature_annealing = use_temperature_annealing
        self.temperature_init = temperature_init
        self.use_cross_entropy_reg = use_cross_entropy_reg
        self.xent_init = xent_init
        self.random_state = random_state
        self.verbose = verbose
        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, obs_dim: int, num_steps: int):
        """Construct the SNLDS model, with initial weights set by ``random_state``.

        Keras 3 draws the seed of an initializer created without one from
        Python's ``random`` module, which ``tf.random.set_seed`` does not
        seed: two fits with the same ``random_state`` started from different
        weights. The module is seeded while the layers are created, then given
        its state back, so the caller's ``random`` stream is left as it was.
        """
        if self.random_state is None:
            return self._construct_model(obs_dim, num_steps)
        state = random.getstate()
        random.seed(self.random_state)
        try:
            return self._construct_model(obs_dim, num_steps)
        finally:
            random.setstate(state)

    def _construct_model(self, obs_dim: int, num_steps: int):
        """Build bidirectional inference and learned Gaussian noise models."""
        import tensorflow as tf

        from .vendor import config_utils, model_cavi_snlds, utils

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
        K, H, R = self.n_states, self.hidden_dim, self.rnn_dim
        rnn_class = {
            "gru": tf.keras.layers.GRU,
            "lstm": tf.keras.layers.LSTM,
            "simplernn": tf.keras.layers.SimpleRNN,
        }[self.rnn_type]
        embedding = tf.keras.layers.Bidirectional(rnn_class(R, return_sequences=True))
        emission = config_utils.get_distribution_config(
            triangular_cov=False, trainable_cov=True, sigma_scale=0.2
        )
        inference = config_utils.get_distribution_config(
            triangular_cov=False, trainable_cov=True, sigma_scale=0.1
        )
        initial = config_utils.get_distribution_config(
            triangular_cov=False, trainable_cov=False, sigma_scale=1.0
        )
        transition = config_utils.get_distribution_config(
            triangular_cov=False, trainable_cov=True, sigma_scale=0.2
        )
        discrete = utils.build_dense_network([max(16, 4 * K), K * K], ["relu", None])
        if K > 1:
            bias = np.eye(K) * np.log(0.98 * (K - 1) / 0.02)
            discrete.layers[-1].bias_initializer = tf.keras.initializers.Constant(
                bias.ravel()
            )
        return model_cavi_snlds.create_model(
            num_categ=K,
            hidden_dim=H,
            observation_dim=obs_dim,
            config_emission=emission,
            config_inference=inference,
            config_z_initial=initial,
            config_z_transition=transition,
            network_emission=utils.build_dense_network([32, obs_dim], ["relu", None]),
            network_input_embedding=embedding,
            network_posterior_mlp=utils.build_dense_network([32, H], ["relu", None]),
            network_posterior_rnn=utils.build_rnn_cell(self.rnn_type, R),
            network_s_transition=discrete,
            networks_z_transition=[
                utils.build_dense_network([64, H], ["relu", None]) for _ in range(K)
            ],
            inference_type=self.inference_type,
        )

    def _get_schedules(self):
        """Temperature and cross-entropy weight as functions of the step.

        The paper's Reacher schedule (300,000 steps) scaled to ``n_train_steps``,
        see the class Notes.
        """
        from .vendor import config_utils, utils

        n = self.n_train_steps
        decay_steps = max(1.0, n / 600.0)

        temp_config = config_utils.get_temperature_config(
            decay_rate=0.975,
            decay_steps=decay_steps,
            initial_temperature=float(self.temperature_init),
            minimal_temperature=1.0,
            kickin_steps=n / 3.0,
            use_temperature_annealing=self.use_temperature_annealing,
        )
        xent_config = config_utils.get_cross_entropy_config(
            decay_rate=0.975,
            decay_steps=decay_steps,
            initial_value=float(self.xent_init),
            kickin_steps=n / 6.0,
            use_entropy_annealing=self.use_cross_entropy_reg,
        )

        def get_temperature(step):
            if temp_config.use_temperature_annealing:
                return utils.schedule_exponential_decay(
                    step, temp_config, temp_config.minimal_temperature
                )
            return 1.0

        def get_xent_coef(step):
            if xent_config.use_entropy_annealing:
                return utils.schedule_exponential_decay(step, xent_config)
            return 0.0

        return get_temperature, get_xent_coef

    # ------------------------------------------------------------------
    # BaseSegmenter interface
    # ------------------------------------------------------------------

    def _features(self, X):
        if self.feature_window == 1:
            return X
        from scipy.ndimage import uniform_filter1d

        X = np.asarray(X, dtype=np.float64)
        mean = uniform_filter1d(X, self.feature_window, axis=0, mode="nearest")
        variance = (
            uniform_filter1d(X * X, self.feature_window, axis=0, mode="nearest")
            - mean * mean
        )
        return np.concatenate([mean, np.sqrt(np.maximum(variance, 0.0))], axis=1)

    def _initialize(self, data, next_batch, seed):
        """Initialize generative modes from an unlabeled latent encoding."""
        import tensorflow as tf
        from sklearn.cluster import KMeans

        model = self._model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        @tf.function(autograph=False)
        def encode(batch):
            return model.inference_network(batch, random_seed=seed, sample=False)[0][0]

        @tf.function(autograph=False)
        def pretrain(batch):
            with tf.GradientTape() as tape:
                z = encode(batch)
                reconstructed = model.x_emit(z).mean()
                loss = tf.reduce_mean(tf.square(reconstructed - batch))
            variables = (
                model.inference_network.trainable_variables
                + model.x_emit.x_emission_net.trainable_variables
            )
            gradients = tape.gradient(loss, variables)
            pairs = [
                (g, v)
                for g, v in zip(gradients, variables, strict=True)
                if g is not None
            ]
            optimizer.apply_gradients(pairs)
            return loss

        for _ in range(self.init_steps):
            pretrain(next_batch())
        means = encode(tf.constant(data[None], tf.float32)).numpy()[0]
        n_clusters = min(self.n_states, len(data))
        clustering = KMeans(
            n_clusters=n_clusters, n_init=10, random_state=self.random_state
        ).fit(means)
        centers = clustering.cluster_centers_[np.arange(self.n_states) % n_clusters]
        model.z_tran(tf.constant(means[None], tf.float32))
        model.s_tran(tf.constant(data[None], tf.float32))
        for k, network in enumerate(model.z_tran.z_trans_networks):
            network.layers[-1].kernel.assign(tf.zeros_like(network.layers[-1].kernel))
            network.layers[-1].bias.assign(centers[k])
        model.z0_dist.loc.assign(centers)
        counts = np.ones((self.n_states, self.n_states)) * 0.01
        counts += np.eye(self.n_states) * 10.0
        labels = clustering.labels_
        np.add.at(counts, (labels[1:], labels[:-1]), 1)
        transition = counts / counts.sum(axis=0, keepdims=True)
        last = model.s_tran.dense_net.layers[-1]
        last.kernel.assign(tf.zeros_like(last.kernel))
        last.bias.assign(np.log(transition).ravel())

    def _fit(self, X: np.ndarray, y=None):
        """Learn a switching model from observations; y is never used."""
        import tensorflow as tf

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        fit_seed = (
            int(self.random_state)
            if self.random_state is not None
            else int(rng.integers(0, 2**31 - 1))
        )
        X = self._features(X)[:: self.downsample]
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        data = ((X - self._mean) / self._std).astype(np.float32)
        n_timepoints, obs_dim = data.shape
        self._obs_dim = obs_dim
        window = (
            n_timepoints
            if self.train_window is None
            else min(self.train_window, n_timepoints)
        )
        offsets = np.arange(window)

        def next_batch():
            starts = rng.integers(0, n_timepoints - window + 1, self.batch_size)
            return tf.constant(data[starts[:, None] + offsets[None, :]])

        self._model = self._build_model(obs_dim, window)
        if self.init_steps:
            self._initialize(data, next_batch, tf.constant([fit_seed, -1], tf.int32))
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        get_temperature, get_xent_coef = self._get_schedules()

        @tf.function(autograph=False)
        def train_step(batch, temperature, xent_coef):
            seed = tf.stack(
                [
                    tf.constant(fit_seed, tf.int32),
                    tf.cast(optimizer.iterations, tf.int32),
                ]
            )
            with tf.GradientTape() as tape:
                result = self._model(batch, temperature=temperature, random_seed=seed)
                likelihood = result[self.objective]
                loss = -likelihood
                if self.use_cross_entropy_reg:
                    loss -= xent_coef * result["cross_entropy"]
                loss /= tf.cast(tf.shape(batch)[1], tf.float32)
                tf.debugging.assert_all_finite(loss, "Non-finite SNLDS objective")
            variables = self._model.trainable_variables
            pairs = [
                (g, v)
                for g, v in zip(tape.gradient(loss, variables), variables, strict=True)
                if g is not None
            ]
            gradients, _ = tf.clip_by_global_norm([g for g, _ in pairs], 5.0)
            optimizer.apply_gradients(
                zip(gradients, [v for _, v in pairs], strict=True)
            )
            return likelihood

        for step in range(self.n_train_steps):
            likelihood = train_step(
                next_batch(),
                tf.constant(get_temperature(step), tf.float32),
                tf.constant(get_xent_coef(step), tf.float32),
            )
            if self.verbose >= 1 and step % max(1, self.n_train_steps // 10) == 0:
                logger.info(
                    "SNLDS step %d/%d: %.4f",
                    step,
                    self.n_train_steps,
                    float(likelihood),
                )
        self._make_inference(fit_seed)
        return self

    def _make_inference(self, fit_seed):
        import tensorflow as tf

        from .vendor import forward_backward_algo

        @tf.function(
            input_signature=[tf.TensorSpec([1, None, self._obs_dim], tf.float32)],
            autograph=False,
        )
        def infer(data):
            seed = (
                tf.constant([fit_seed, 0], tf.int32)
                if self.random_state is not None
                else None
            )
            z = self._model.inference_network(
                data, random_seed=seed, sample=not self.posterior_mean
            )[0][0]
            log_b, log_a = self._model.calculate_likelihoods(data, z)
            if self.decoding == "viterbi":
                return log_a[0], log_b[0]
            return forward_backward_algo.forward_backward(
                log_a, log_b, self._model.log_init
            )[2][0]

        self._infer = infer

    def _predict(self, X: np.ndarray) -> np.ndarray:
        import tensorflow as tf

        from .decoding import viterbi

        n_timepoints = len(X)
        data = self._features(X)[:: self.downsample]
        data = ((data - self._mean) / self._std).astype(np.float32)
        result = self._infer(tf.constant(data[None]))
        if self.decoding == "viterbi":
            labels = viterbi(
                result[0].numpy(), result[1].numpy(), self._model.log_init.numpy()
            )
        else:
            labels = np.argmax(result.numpy(), axis=-1).astype(np.int64)
        return np.repeat(labels, self.downsample)[:n_timepoints]
