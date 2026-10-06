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
from typing import Any

import numpy as np

from ..base import BaseSegmenter

logger = logging.getLogger(__name__)


class SNLDSDetector(BaseSegmenter):
    """State detector based on Switching Non-Linear Dynamical Systems (CAVI-SNLDS).

    Uses collapsed amortized variational inference to learn discrete regimes
    in time series data.  After training, the discrete state posterior
    ``p(s_t | x_{1:T}, z_{1:T})`` is used to assign a regime label to each
    time-step.

    Parameters
    ----------
    n_states : int, default=3
        Number of discrete regimes (K).
    hidden_dim : int, default=8
        Dimension of the continuous latent state z_t.
    rnn_dim : int, default=4
        Hidden dimension of the posterior inference RNN.
    rnn_type : str, default="simplernn"
        RNN cell type for inference network ("gru", "lstm", "simplernn").
    n_train_steps : int, default=5000
        Number of gradient steps for training.
    learning_rate : float, default=1e-4
        Learning rate for Adam optimiser.
    batch_size : int, default=1
        Number of sequences per mini-batch (kept at 1 for single-series usage).
    objective : str, default="elbo"
        Training objective ("elbo" or "iwae").
    use_temperature_annealing : bool, default=True
        Whether to anneal the discrete state temperature. Currently inactive, see
        Notes.
    use_cross_entropy_reg : bool, default=True
        Whether to use cross-entropy regularisation to encourage state usage.
        Currently inactive, see Notes.
    random_state : int or None, default=42
        Random seed for reproducibility.
    verbose : int, default=0
        Logging verbosity (0=silent, 1=progress, 2=debug).
    axis : int, default=0
        Time axis of the input array.

    Notes
    -----
    The schedules take the values of the upstream Lorenz example, whose
    initial temperature and initial regulariser weight are 0. The temperature
    therefore stays at 1 and the regulariser weight at 1e-10 for the whole
    training, whatever ``use_temperature_annealing`` and
    ``use_cross_entropy_reg``. The paper starts both large and decays them to
    prevent the collapse onto a single regime.

    The vendored model code comes from Google Research (Apache License 2.0),
    see ``tsseg/algorithms/snlds/vendor``.

    References
    ----------
    .. [1] Dong, Z., Seybold, B., Murphy, K., & Bui, H. (2020). Collapsed
       Amortized Variational Inference for Switching Nonlinear Dynamical
       Systems. ICML.
    """

    _tags: dict[str, Any] = {
        "X_inner_type": "np.ndarray",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        # n_states is the number of regimes of the model, not an upper bound: the
        # cross-entropy regulariser pushes the posterior to use all of them.
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
        "python_dependencies": ["tensorflow", "tensorflow_probability", "tf_keras"],
    }

    def __init__(
        self,
        *,
        n_states: int = 3,
        hidden_dim: int = 8,
        rnn_dim: int = 4,
        rnn_type: str = "simplernn",
        n_train_steps: int = 5000,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        objective: str = "elbo",
        use_temperature_annealing: bool = True,
        use_cross_entropy_reg: bool = True,
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
        self.objective = objective
        self.use_temperature_annealing = use_temperature_annealing
        self.use_cross_entropy_reg = use_cross_entropy_reg
        self.random_state = random_state
        self.verbose = verbose
        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, obs_dim: int, num_steps: int):
        """Construct the SNLDS model and training utilities."""
        import tensorflow as tf

        from .vendor import config_utils, model_cavi_snlds, utils

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)

        K = self.n_states
        H = self.hidden_dim

        # --- Emission p(x_t | z_t) ---
        config_emission = config_utils.get_distribution_config(
            triangular_cov=False,
            trainable_cov=False,
        )
        emission_network = utils.build_dense_network(
            [max(8, H), 32, obs_dim],
            ["relu", "relu", None],
        )

        # --- Inference q(z_t | x_{1:T}) ---
        config_inference = config_utils.get_distribution_config(triangular_cov=True)
        posterior_rnn = utils.build_rnn_cell(
            rnn_type=self.rnn_type,
            rnn_hidden_dim=self.rnn_dim,
        )
        posterior_mlp = utils.build_dense_network(
            [32, H],
            ["relu", None],
        )

        # --- Initial distribution p(z_0) ---
        config_z_initial = config_utils.get_distribution_config(triangular_cov=True)

        # --- Continuous transition p(z_t | z_{t-1}, s_t) ---
        config_z_transition = config_utils.get_distribution_config(
            triangular_cov=True,
            trainable_cov=True,
            sigma_scale=0.1,
            raw_sigma_bias=1e-5,
            sigma_min=1e-5,
        )
        z_transition_networks = [
            utils.build_dense_network([256, H], ["relu", None]) for _ in range(K)
        ]

        # --- Discrete transition p(s_t | s_{t-1}, x_{t-1}) ---
        num_categ_sq = K * K
        network_s_transition = utils.build_dense_network(
            [4 * num_categ_sq, num_categ_sq],
            ["relu", None],
        )

        model = model_cavi_snlds.create_model(
            num_categ=K,
            hidden_dim=H,
            observation_dim=obs_dim,
            config_emission=config_emission,
            config_inference=config_inference,
            config_z_initial=config_z_initial,
            config_z_transition=config_z_transition,
            network_emission=emission_network,
            network_input_embedding=lambda x: x,
            network_posterior_mlp=posterior_mlp,
            network_posterior_rnn=posterior_rnn,
            network_s_transition=network_s_transition,
            networks_z_transition=z_transition_networks,
            name="snlds",
        )
        model.build(input_shape=(self.batch_size, num_steps, obs_dim))
        return model

    def _get_schedules(self):
        """Return schedule functions for learning rate, temperature, and cross-entropy."""
        from .vendor import config_utils, utils

        temp_config = config_utils.get_temperature_config(
            decay_rate=0.99,
            decay_steps=50,
            initial_temperature=0.0,
            minimal_temperature=1.0,
            kickin_steps=0,
            use_temperature_annealing=self.use_temperature_annealing,
        )

        xent_config = config_utils.get_cross_entropy_config(
            decay_rate=0.99,
            decay_steps=50,
            initial_value=0.0,
            kickin_steps=0,
            use_entropy_annealing=self.use_cross_entropy_reg,
        )

        def get_temperature(step):
            if temp_config.use_temperature_annealing:
                return utils.schedule_exponential_decay(
                    step,
                    temp_config,
                    temp_config.minimal_temperature,
                )
            return temp_config.initial_temperature

        def get_xent_coef(step):
            if xent_config.use_entropy_annealing:
                return utils.schedule_exponential_decay(step, xent_config)
            return 0.0

        return get_temperature, get_xent_coef

    # ------------------------------------------------------------------
    # BaseSegmenter interface
    # ------------------------------------------------------------------

    def _fit(self, X: np.ndarray, y=None):
        """Train the SNLDS model on the input time series.

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_channels)
        """
        import tensorflow as tf

        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

        n_timepoints, obs_dim = X.shape
        self._obs_dim = obs_dim

        # Normalise input
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Build model
        self._model = self._build_model(obs_dim, n_timepoints)
        optimizer = tf.keras.optimizers.Adam()

        get_temperature, get_xent_coef = self._get_schedules()

        # Prepare batch: [1, T, D]
        X_tensor = tf.constant(X_norm[np.newaxis, :, :], dtype=tf.float32)

        # Compiled training step for performance
        objective_key = self.objective

        @tf.function
        def _train_step(data, temperature, xent_coef):
            with tf.GradientTape() as tape:
                result = self._model(data, temperature, num_samples=1)
                log_likelihood = result[objective_key]
                cross_entropy = result["cross_entropy"]
                loss = -1.0 * (log_likelihood + xent_coef * cross_entropy)

            grads = tape.gradient(loss, self._model.trainable_variables)
            clipped_grads = [
                tf.clip_by_value(g, -5.0, 5.0) if g is not None else g for g in grads
            ]
            grads_and_vars = [
                (g, v)
                for g, v in zip(
                    clipped_grads, self._model.trainable_variables, strict=True
                )
                if g is not None
            ]
            optimizer.apply_gradients(grads_and_vars)
            return log_likelihood, cross_entropy

        # Training loop
        for step in range(self.n_train_steps):
            temperature = get_temperature(step)
            xent_coef = get_xent_coef(step)
            optimizer.learning_rate = self.learning_rate

            log_likelihood, _ = _train_step(
                X_tensor,
                tf.constant(temperature, dtype=tf.float32),
                tf.constant(xent_coef, dtype=tf.float32),
            )

            if self.verbose >= 1 and (step % max(1, self.n_train_steps // 10)) == 0:
                ll_val = tf.reduce_mean(log_likelihood).numpy()
                logger.info(
                    "SNLDS step %d/%d — %s: %.4f",
                    step,
                    self.n_train_steps,
                    self.objective,
                    ll_val,
                )

        # Store normalised data for predict
        self._X_norm = X_norm
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict discrete state labels for each time step.

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_channels)

        Returns
        -------
        labels : np.ndarray, shape (n_timepoints,)
            Integer state labels in {0, ..., n_states-1}.
        """
        import tensorflow as tf

        # Normalise with training stats
        X_norm = (X - self._mean) / self._std
        X_tensor = tf.constant(X_norm[np.newaxis, :, :], dtype=tf.float32)

        # Run model at temperature=1.0 (no annealing at inference)
        result = self._model(X_tensor, temperature=1.0, num_samples=1)

        # posterior_llk: [1, T, K] — log p(s_t = k | x, z)
        log_posterior = result["posterior_llk"]
        labels = tf.argmax(log_posterior, axis=-1).numpy()[0]  # [T]

        return labels.astype(np.int64)
