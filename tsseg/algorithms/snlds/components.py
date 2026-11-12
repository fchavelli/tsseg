"""Positive Gaussian parameters and inference components implemented in tsseg."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


def stateless_seed(seed):
    """Return a two-integer seed; draw one fresh seed when none is supplied."""
    if seed is None:
        return tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32)
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)
    if seed.shape.rank == 0:
        return tf.stack([seed, tf.constant(0, tf.int32)])
    return tf.ensure_shape(seed, [2])


class GaussianScale(tf.keras.layers.Layer):
    """A positive Gaussian scale with a learnable magnitude."""

    def __init__(
        self,
        dimension,
        triangular,
        trainable,
        raw_sigma_bias,
        sigma_min,
        sigma_scale,
        dtype,
        name,
    ):
        super().__init__(dtype=dtype, name=name)
        self.triangular = triangular
        self.raw_sigma_bias = raw_sigma_bias
        self.sigma_min = sigma_min
        self.sigma_scale = sigma_scale
        diagonal = tf.fill([dimension], tf.cast(0.541324854612918, dtype))
        initial = (
            tfp.math.fill_triangular_inverse(tf.linalg.diag(diagonal))
            if triangular
            else diagonal
        )
        self.raw_cov = self.add_weight(
            name="raw_cov",
            shape=initial.shape,
            dtype=dtype,
            trainable=trainable,
            initializer=tf.keras.initializers.Constant(initial),
        )
        self.built = True

    @property
    def value(self):
        raw = self.raw_cov
        if self.triangular:
            raw = tfp.math.fill_triangular(raw)
            diagonal = (
                tf.nn.softplus(tf.linalg.diag_part(raw) + self.raw_sigma_bias)
                * self.sigma_scale
                + self.sigma_min
            )
            return tf.linalg.set_diag(raw * self.sigma_scale, diagonal)
        return (
            tf.nn.softplus(raw + self.raw_sigma_bias) * self.sigma_scale
            + self.sigma_min
        )


class InitialStatePrior(tf.keras.layers.Layer):
    """Tracked Gaussian initial means and a shared positive covariance scale."""

    def __init__(
        self,
        latent_dim,
        num_categ,
        use_trainable_cov=False,
        use_triangular_cov=False,
        raw_sigma_bias=0.0,
        sigma_min=1e-5,
        sigma_scale=1.0,
        dtype=tf.float32,
        name="z0",
    ):
        super().__init__(dtype=dtype, name=name)
        self.loc = self.add_weight(
            name="initial_mean",
            shape=[num_categ, latent_dim],
            dtype=dtype,
            initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.triangular = use_triangular_cov
        self.scale = GaussianScale(
            latent_dim,
            use_triangular_cov,
            use_trainable_cov,
            raw_sigma_bias,
            sigma_min,
            sigma_scale,
            dtype,
            name + "_scale",
        )

    def call(self, samples):
        if self.triangular:
            distribution = tfp.distributions.MultivariateNormalTriL(
                self.loc, scale_tril=self.scale.value
            )
        else:
            distribution = tfp.distributions.MultivariateNormalDiag(
                self.loc, scale_diag=self.scale.value
            )
        return distribution.log_prob(samples)

    def log_prob(self, samples):
        return self(samples)


class FactorizedInference(tf.keras.layers.Layer):
    """Independent latent noise conditional on a bidirectional observation RNN.

    This is a restricted posterior family, rather than the paper's
    autoregressive continuous posterior.
    """

    def __init__(self, posterior_dist, embedding_network):
        super().__init__()
        self.posterior_dist = posterior_dist
        self.embedding_network = embedding_network

    def call(self, inputs, num_samples=1, random_seed=None, sample=True):
        distribution = self.posterior_dist(self.embedding_network(inputs))
        if sample:
            z = distribution.sample(num_samples, seed=stateless_seed(random_seed))
        else:
            z = tf.tile(distribution.mean()[None], [num_samples, 1, 1, 1])
        entropy = tf.broadcast_to(distribution.entropy()[None], tf.shape(z)[:-1])
        return z, entropy, distribution.log_prob(z)
