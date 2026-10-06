"""Tests of GaussianHMMDetector: EM and Viterbi against hmmlearn, and behaviour."""

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from tsseg.algorithms import GaussianHMMDetector
from tsseg.algorithms.gaussian_hmm import _core
from tsseg.algorithms.gaussian_hmm.detector import _diag_log_density, _log


def _regimes(rng=0, d=2):
    """Five segments over three Gaussian regimes, 1,500 points."""
    rng = np.random.default_rng(rng)
    spec = [(0, 1), (3, 0.5), (-2, 2), (3, 0.5), (0, 1)]
    X = np.concatenate([rng.normal(m, s, (300, d)) for m, s in spec])
    y = np.repeat([0, 1, 2, 1, 0], 300)
    return X, y


def _random_start(X, K, rng):
    startprob = np.full(K, 1 / K)
    transmat = np.full((K, K), 1 / K)
    means = X[rng.choice(len(X), K, replace=False)]
    variances = np.tile(X.var(axis=0), (K, 1))
    return startprob, transmat, means, variances


# ---------------------------------------------------------------------------
# Reference implementation: hmmlearn (BSD-3), skipped when it is not installed
# ---------------------------------------------------------------------------


def _hmmlearn_model(K, n_iter, params):
    hmm = pytest.importorskip("hmmlearn.hmm")
    model = hmm.GaussianHMM(
        K,
        covariance_type="diag",
        n_iter=n_iter,
        tol=-np.inf,
        init_params="",
        params="stmc",
        covars_prior=0.0,
        implementation="log",
    )
    model.startprob_, model.transmat_, means, variances = (np.copy(p) for p in params)
    model.means_, model.covars_ = means, variances
    return model


def test_baum_welch_matches_hmmlearn():
    """Same start, same number of iterations: same parameters and likelihood."""
    X, _ = _regimes()
    params = _random_start(X, 3, np.random.default_rng(1))
    ref = _hmmlearn_model(3, 25, params).fit(X)

    det = GaussianHMMDetector(n_states=3, n_iter=25, reg_covar=0.0)
    det.tol = -np.inf  # run exactly n_iter iterations, as the reference
    (sp, tm, mu, var), ll = det._em(X, params)

    np.testing.assert_allclose(sp, ref.startprob_, atol=1e-10)
    np.testing.assert_allclose(tm, ref.transmat_, atol=1e-10)
    np.testing.assert_allclose(mu, ref.means_, atol=1e-10)
    np.testing.assert_allclose(var, ref._covars_, atol=1e-10)
    np.testing.assert_allclose(ll, ref.score(X), rtol=1e-10)


def test_viterbi_matches_hmmlearn():
    X, _ = _regimes(rng=2)
    params = _random_start(X, 3, np.random.default_rng(3))
    ref = _hmmlearn_model(3, 10, params).fit(X)
    log_b = _diag_log_density(X, ref.means_, ref._covars_)
    path = _core.viterbi(log_b, _log(ref.startprob_), _log(ref.transmat_))
    ref_logprob, ref_path = ref.decode(X, algorithm="viterbi")
    np.testing.assert_array_equal(path, ref_path)


def test_mocap_matches_hmmlearn():
    """Agreement with hmmlearn on real data: MoCap trial 0, same start, 4 states."""
    from tsseg.data.datasets import load_mocap

    X, _ = load_mocap(trial=0)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    det = GaussianHMMDetector(n_states=4, n_iter=50, reg_covar=0.0, random_state=0)
    det.tol = -np.inf
    params = det._init_params(X, 4, seed=0)
    ref = _hmmlearn_model(4, 50, params).fit(X)
    (sp, tm, mu, var), ll = det._em(X, params)
    np.testing.assert_allclose(ll, ref.score(X), rtol=1e-8)
    np.testing.assert_allclose(tm, ref.transmat_, atol=1e-7)
    np.testing.assert_allclose(mu, ref.means_, atol=1e-7)
    np.testing.assert_allclose(var, ref._covars_, atol=1e-7)
    log_b = _diag_log_density(X, mu, var)
    path = _core.viterbi(log_b, _log(sp), _log(tm))
    assert adjusted_rand_score(path, ref.predict(X)) == 1.0


# ---------------------------------------------------------------------------
# Forward-backward on a model small enough to enumerate
# ---------------------------------------------------------------------------


def test_forward_backward_against_enumeration():
    rng = np.random.default_rng(4)
    n, K = 5, 2
    log_b = rng.normal(size=(n, K))
    pi = np.array([0.3, 0.7])
    A = np.array([[0.8, 0.2], [0.4, 0.6]])
    gamma, xi_sum, ll = _core.forward_backward(log_b, np.log(pi), np.log(A))

    paths = np.array(np.meshgrid(*[range(K)] * n, indexing="ij")).reshape(n, -1).T
    joint = np.array(
        [
            np.log(pi[p[0]])
            + sum(np.log(A[p[t - 1], p[t]]) for t in range(1, n))
            + sum(log_b[t, p[t]] for t in range(n))
            for p in paths
        ]
    )
    post = np.exp(joint - np.logaddexp.reduce(joint))
    np.testing.assert_allclose(ll, np.logaddexp.reduce(joint), rtol=1e-12)
    for t in range(n):
        for k in range(K):
            np.testing.assert_allclose(gamma[t, k], post[paths[:, t] == k].sum())
    expected_xi = np.zeros((K, K))
    for p, w in zip(paths, post, strict=True):
        for t in range(n - 1):
            expected_xi[p[t], p[t + 1]] += w
    np.testing.assert_allclose(xi_sum, expected_xi)
    best = paths[np.argmax(joint)]
    np.testing.assert_array_equal(_core.viterbi(log_b, np.log(pi), np.log(A)), best)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def test_guided_recovers_regimes():
    X, y = _regimes()
    labels = GaussianHMMDetector(n_states=3, random_state=0).fit_predict(X)
    assert adjusted_rand_score(y, labels) > 0.95


def test_bic_selects_the_number_of_states():
    X, y = _regimes()
    det = GaussianHMMDetector(random_state=0, max_states=6).fit(X)
    assert det.n_states_ == 3
    assert set(det.bic_) == set(range(1, 7))
    assert adjusted_rand_score(y, det.predict(X)) > 0.95


def test_bic_selects_one_state_on_noise():
    X = np.random.default_rng(5).normal(size=(1000, 2))
    det = GaussianHMMDetector(random_state=0, max_states=4).fit(X)
    assert det.n_states_ == 1
    assert np.all(det.predict(X) == 0)


def test_single_state_returns_constant_labels():
    X, _ = _regimes()
    labels = GaussianHMMDetector(n_states=1).fit_predict(X)
    assert labels.shape == (len(X),)
    assert np.all(labels == 0)


def test_labels_numbered_by_first_appearance():
    X, _ = _regimes()
    labels = GaussianHMMDetector(n_states=3, random_state=0).fit_predict(X)
    _, first = np.unique(labels, return_index=True)
    assert labels[0] == 0
    assert np.all(np.diff(labels[np.sort(first)]) == 1)


def test_reproducible_with_seed():
    X, _ = _regimes(rng=6)
    a = GaussianHMMDetector(n_states=3, random_state=7).fit_predict(X)
    b = GaussianHMMDetector(n_states=3, random_state=7).fit_predict(X)
    np.testing.assert_array_equal(a, b)


def test_univariate_input():
    X, y = _regimes(d=1)
    labels = GaussianHMMDetector(n_states=3, random_state=0).fit_predict(X[:, 0])
    assert adjusted_rand_score(y, labels) > 0.9


def test_n_states_larger_than_series_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        GaussianHMMDetector(n_states=20).fit(np.zeros((10, 1)))


def test_reg_covar_must_be_positive():
    with pytest.raises(ValueError, match="reg_covar"):
        GaussianHMMDetector(n_states=2, reg_covar=0.0).fit(np.zeros((10, 1)))


def test_unnormalised_data_with_large_offset():
    """Centred variance and density: no cancellation when normalize=False."""
    X, y = _regimes()
    labels = GaussianHMMDetector(
        n_states=3, normalize=False, random_state=0
    ).fit_predict(X + 1e8)
    assert adjusted_rand_score(y, labels) > 0.95


def test_predict_rejects_other_channel_count():
    X, _ = _regimes(d=2)
    det = GaussianHMMDetector(n_states=3, random_state=0).fit(X)
    with pytest.raises(ValueError, match="channels"):
        det.predict(X[:, 0])


def test_predict_on_another_series():
    X, _ = _regimes(rng=0)
    X2, y2 = _regimes(rng=8)
    det = GaussianHMMDetector(n_states=3, random_state=0).fit(X)
    assert adjusted_rand_score(y2, det.predict(X2)) > 0.95


def test_tol_stops_em_early():
    X, _ = _regimes()
    loose = GaussianHMMDetector(n_states=3, tol=1.0, random_state=0).fit(X)
    tight = GaussianHMMDetector(n_states=3, tol=0.0, n_iter=60, random_state=0).fit(X)
    assert loose.n_iter_ < tight.n_iter_


def test_warns_when_bic_reaches_max_states():
    X, _ = _regimes()
    with pytest.warns(UserWarning, match="max_states=2"):
        GaussianHMMDetector(max_states=2, random_state=0).fit(X)
