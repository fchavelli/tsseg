"""Numerical contracts of the vendored SNLDS model, independent of fit quality."""

import itertools

import numpy as np
import pytest

from .test_snlds import _tf_stack


def test_expected_complete_likelihood_matches_enumerated_state_paths():
    """A transition's emission belongs to its destination state, not its source."""
    _tf_stack()
    import tensorflow as tf

    from tsseg.algorithms.snlds.vendor import forward_backward_algo, model_base

    log_a = tf.math.log(
        [[[[0.5, 0.5], [0.5, 0.5]], [[0.1, 0.8], [0.9, 0.2]], [[0.7, 0.3], [0.3, 0.7]]]]
    )
    log_b = tf.math.log([[[0.8, 0.2], [0.05, 0.95], [0.9, 0.1]]])
    log_init = tf.math.log([0.6, 0.4])
    _, _, gamma1, gamma2 = forward_backward_algo.forward_backward(
        log_a, log_b, log_init
    )
    # Independently enumerate every path of this tiny HMM.
    a, b, pi = log_a.numpy()[0], log_b.numpy()[0], log_init.numpy()
    joints = np.array(
        [
            pi[path[0]]
            + b[0, path[0]]
            + sum(a[t, path[t], path[t - 1]] + b[t, path[t]] for t in (1, 2))
            for path in itertools.product(range(2), repeat=3)
        ]
    )
    posterior = np.exp(joints - np.logaddexp.reduce(joints))
    # The helper has no dependence on model state.
    seq, initial = model_base.SwitchingNLDS._get_log_likelihood(
        None, log_a, log_b, log_init, gamma1, gamma2
    )
    assert float((seq + initial)[0]) == pytest.approx(
        np.dot(posterior, joints), abs=1e-6
    )


@pytest.mark.parametrize("reuse_forward_likelihood", [False, True])
def test_collapsed_objectives_and_gradients_match_enumerated_state_paths(
    reuse_forward_likelihood,
):
    """Marginalising the discrete paths includes their posterior entropy."""
    _tf_stack()
    import tensorflow as tf

    from tsseg.algorithms.snlds.vendor import forward_backward_algo, model_base

    model = model_base.SwitchingNLDS(
        None, None, None, None, None, continuous_state_dim=1, num_categories=2
    )
    log_a = tf.math.log(
        [[[[0.5, 0.5], [0.5, 0.5]], [[0.1, 0.8], [0.9, 0.2]], [[0.7, 0.3], [0.3, 0.7]]]]
    )
    log_b = tf.Variable(tf.math.log([[[0.8, 0.2], [0.05, 0.95], [0.9, 0.1]]]))
    log_init = tf.math.log([0.6, 0.4])
    with tf.GradientTape(persistent=True) as tape:
        _, _, gamma1, gamma2, log_marginal = forward_backward_algo.forward_backward(
            log_a, log_b, log_init, return_log_likelihood=True
        )
        result = model.get_objective_values(
            log_a,
            log_b,
            log_init,
            gamma1,
            gamma2,
            tf.zeros((1, 3)),
            tf.zeros((1, 3)),
            num_samples=1,
            log_marginal=log_marginal if reuse_forward_likelihood else None,
        )
        joints = tf.stack(
            [
                log_init[path[0]]
                + log_b[0, 0, path[0]]
                + sum(
                    log_a[0, t, path[t], path[t - 1]] + log_b[0, t, path[t]]
                    for t in (1, 2)
                )
                for path in itertools.product(range(2), repeat=3)
            ]
        )
        exact = tf.reduce_logsumexp(joints)
    for objective in ("elbo", "iwae"):
        np.testing.assert_allclose(result[objective], exact, atol=1e-6)
        np.testing.assert_allclose(
            tape.gradient(result[objective], log_b),
            tape.gradient(exact, log_b),
            atol=1e-6,
        )


@pytest.mark.parametrize("triangular", [False, True])
@pytest.mark.parametrize("kind", ["transition", "emission"])
def test_trainable_scale_has_a_live_gradient(kind, triangular):
    """Covariances must remain tracked and be recomputed inside GradientTape."""
    _tf_stack()
    import tensorflow as tf

    from tsseg.algorithms.snlds.vendor import model_cavi_snlds as core

    if kind == "transition":
        layer = core.ContinuousStateTransition(
            [tf.keras.layers.Dense(2)], 2, use_triangular_cov=triangular
        )
        sample = tf.ones((1, 3, 1, 2))
    else:
        layer = core.GaussianDistributionFromMean(
            tf.keras.layers.Dense(2), 2, use_triangular_cov=triangular
        )
        sample = tf.ones((1, 3, 2))
    layer(tf.zeros((1, 3, 2)))
    covariance_weights = [v for v in layer.trainable_variables if "cov" in v.name]
    assert len(covariance_weights) == 1

    @tf.function
    def update():
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(layer(tf.zeros((1, 3, 2))).log_prob(sample))
        gradient = tape.gradient(loss, covariance_weights)[0]
        assert gradient is not None
        covariance_weights[0].assign_sub(1e-6 * gradient)
        return gradient

    before = layer.cov_mat.numpy().copy()
    gradient = update().numpy()
    assert np.isfinite(gradient).all()
    assert np.any(gradient != 0)
    assert np.any(before != layer.cov_mat.numpy())


@pytest.mark.parametrize("triangular", [False, True])
def test_fixed_scale_is_tracked_without_becoming_trainable(triangular):
    _tf_stack()
    import tensorflow as tf

    from tsseg.algorithms.snlds.vendor.model_cavi_snlds import (
        GaussianDistributionFromMean,
    )

    layer = GaussianDistributionFromMean(
        tf.keras.layers.Dense(2),
        2,
        use_triangular_cov=triangular,
        use_trainable_cov=False,
    )
    layer(tf.zeros((1, 3, 2)))
    assert not any("cov" in v.name for v in layer.trainable_variables)
    weights = [v for v in layer.non_trainable_variables if "cov" in v.name]
    assert len(weights) == 1
    before = layer.cov_mat.numpy().copy()
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(layer(tf.zeros((1, 3, 2))).log_prob(tf.ones((1, 3, 2))))
    gradients = tape.gradient(loss, layer.trainable_variables)
    tf.keras.optimizers.SGD(1e-6).apply_gradients(
        zip(gradients, layer.trainable_variables, strict=True)
    )
    np.testing.assert_array_equal(before, layer.cov_mat.numpy())


@pytest.mark.parametrize("rnn_type", ["simplernn", "gru", "lstm"])
def test_all_advertised_rnn_types_fit_and_predict(rnn_type):
    _tf_stack()
    from tsseg.algorithms import SNLDSDetector

    X = np.random.default_rng(0).normal(size=(24, 1))
    result = SNLDSDetector(
        n_states=2,
        rnn_type=rnn_type,
        n_train_steps=1,
        train_window=8,
        batch_size=2,
    ).fit_predict(X)
    assert result.shape == (24,)
