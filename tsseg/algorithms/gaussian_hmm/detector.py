"""Gaussian hidden Markov model fitted by Baum-Welch (EM) and decoded by Viterbi."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from ..base import BaseSegmenter
from ..param_schema import Closed, DataDependent, Interval, ParamDef
from ._core import _HAS_NUMBA, forward_backward, viterbi

__all__ = ["GaussianHMMDetector"]

_LOG2PI = np.log(2.0 * np.pi)


def _diag_log_density(X, means, variances):
    """``log N(x_t; mu_k, diag(var_k))`` for every time step and state, ``(n, K)``."""
    n, K = X.shape[0], means.shape[0]
    out = np.empty((n, K))
    const = X.shape[1] * _LOG2PI + np.sum(np.log(variances), axis=1)
    for k in range(K):
        out[:, k] = -0.5 * (
            const[k] + np.sum((X - means[k]) ** 2 / variances[k], axis=1)
        )
    return out


def _log(a):
    with np.errstate(divide="ignore"):
        return np.log(a)


def _m_step(X, gamma, xi_sum, reg_covar):
    """Re-estimate all parameters (Rabiner, 1989, eqs. 40a-b, 53 and 54).

    ``reg_covar`` is added to the maximum-likelihood variances. A row of the
    transition matrix with no expected transition (a state never visited before
    the last step) is set to uniform.
    """
    startprob = gamma[0] / gamma[0].sum()
    rows = xi_sum.sum(axis=1, keepdims=True)
    K = xi_sum.shape[0]
    transmat = np.where(rows > 0, xi_sum / np.where(rows > 0, rows, 1.0), 1.0 / K)
    weights = gamma.sum(axis=0) + 10 * np.finfo(float).eps
    means = (gamma.T @ X) / weights[:, None]
    variances = np.empty_like(means)
    for k in range(K):
        variances[k] = gamma[:, k] @ (X - means[k]) ** 2 / weights[k]
    return startprob, transmat, means, variances + reg_covar


class GaussianHMMDetector(BaseSegmenter):
    """State detection with a Gaussian hidden Markov model.

    Each hidden state emits a Gaussian with diagonal covariance. The parameters
    (initial and transition probabilities, means, variances) are estimated by
    maximum likelihood with the Baum-Welch algorithm, an expectation-maximisation
    procedure whose E-step is the forward-backward algorithm [1]_. Each time step
    then gets the state of the most probable path (Viterbi).

    With ``n_states=None`` the number of states is chosen by the Bayesian
    information criterion [2]_ among ``1, ..., max_states``.

    Parameters
    ----------
    n_states : int or None, default=None
        Number of hidden states. ``None`` selects it by BIC.
    max_states : int, default=8
        Largest number of states tried when ``n_states`` is ``None``.
    n_iter : int, default=100
        Maximum number of EM iterations per initialisation.
    tol : float, default=1e-4
        EM stops when the log-likelihood per time step improves by less than
        ``tol``.
    n_init : int, default=3
        Number of initialisations (k-means with different seeds); the fit with
        the highest log-likelihood is kept.
    reg_covar : float, default=1e-3
        Added to every variance at each M-step (must be positive), which limits
        the collapse of a state onto a few points. Expressed in units of the
        (normalised) data.
    normalize : bool, default=True
        Z-normalise each channel before fitting.
    random_state : int or None, default=None
        Seed of the k-means initialisations. Pass an int for reproducible
        results.
    axis : int, default=0
        Time axis.

    Attributes
    ----------
    n_states_ : int
        Number of states :math:`K` of the fitted model.
    startprob_ : ndarray of shape (K,)
        Initial state probabilities.
    transmat_ : ndarray of shape (K, K)
        ``transmat_[i, j]`` is the probability of moving from state ``i`` to ``j``.
    means_ : ndarray of shape (K, n_channels)
        Emission means.
    variances_ : ndarray of shape (K, n_channels)
        Emission variances.
    loglik_ : float
        Log-likelihood of the fitted model.
    bic_ : dict
        BIC of each candidate number of states (only when ``n_states=None``).
    n_iter_ : int
        Number of EM iterations of the kept initialisation.

    ``means_``, ``variances_``, ``loglik_`` and ``bic_`` refer to the normalised
    data when ``normalize=True``. The labels returned by ``predict`` are numbered
    by order of first appearance, so label ``k`` is not necessarily row ``k`` of
    these arrays.

    Notes
    -----
    One EM iteration costs :math:`O(nK^2 + nKd)` for :math:`n` time steps,
    :math:`K` states and :math:`d` channels. Selecting :math:`K` by BIC
    multiplies this by the number of candidates. The forward-backward and
    Viterbi recursions use numba when it is installed (``tsseg[accelerators]``)
    and run, much slower, in pure Python otherwise.

    The emissions are i.i.d. given the state, so this model has no notion of
    duration or of temporal shape within a state: it suits regimes that differ in
    level or spread, not regimes that differ only in frequency or waveform. On
    oscillating signals, the states follow the oscillation and the labels change
    within a regime. For the same reason, BIC often selects ``max_states`` on
    long autocorrelated series; a warning says so.

    On integer or coarsely quantised data, a state can collapse onto a single
    value, with a variance of ``reg_covar``; raise ``reg_covar`` in that case.

    The initial probabilities are estimated from one series and are often close
    to one-hot: on another series that starts in a different regime, the first
    labels can be wrong. A state that receives no posterior mass during EM stays
    empty, so fewer than ``n_states`` labels may appear. ``reg_covar`` is added to
    the variances rather than used as a floor, so EM is not strictly monotone;
    a decrease of the log-likelihood stops it.

    References
    ----------
    .. [1] Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
       applications in speech recognition. Proceedings of the IEEE, 77(2),
       257-286.
    .. [2] Schwarz, G. (1978). Estimating the dimension of a model. The Annals of
       Statistics, 6(2), 461-464.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms import GaussianHMMDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 1, 200)])
    >>> labels = GaussianHMMDetector(n_states=2, random_state=0).fit_predict(X)
    >>> int(labels[:200].mean() != labels[200:].mean())
    1
    """

    _tags: dict[str, Any] = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "n_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            nullable=True,
            description="Number of hidden states (None: chosen by BIC).",
        ),
        "max_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Largest number of states tried by BIC.",
        ),
        "n_iter": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Maximum number of EM iterations.",
            group="training",
        ),
        "tol": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Convergence threshold on the log-likelihood per time step.",
            group="training",
        ),
        "n_init": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of k-means initialisations.",
            group="training",
        ),
        "reg_covar": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Value added to the variances at each M-step.",
            group="training",
        ),
        "_cross_constraints": [
            DataDependent(
                "n_states is None or n_states <= n_samples",
                "n_states cannot exceed the number of time steps.",
            ),
        ],
    }

    def __init__(
        self,
        *,
        n_states: int | None = None,
        max_states: int = 8,
        n_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 3,
        reg_covar: float = 1e-3,
        normalize: bool = True,
        random_state: int | None = None,
        axis: int = 0,
    ):
        self.n_states = n_states
        self.max_states = max_states
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.normalize = normalize
        self.random_state = random_state
        super().__init__(axis=axis)

    # ------------------------------------------------------------------ helpers

    def _prepare(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if self.normalize:
            X = (X - self._mean) / self._scale
        return np.ascontiguousarray(X)

    def _init_params(self, X, K, seed):
        km = KMeans(n_clusters=K, n_init=1, random_state=seed).fit(X)
        means = km.cluster_centers_.copy()
        variances = np.tile(X.var(axis=0), (K, 1))
        for k in range(K):
            members = X[km.labels_ == k]
            if len(members) > 1:
                variances[k] = members.var(axis=0)
        variances += self.reg_covar
        startprob = np.full(K, 1.0 / K)
        transmat = np.full((K, K), 1.0 / K)
        return startprob, transmat, means, variances

    def _em(self, X, params):
        """Baum-Welch from ``params``; returns the final parameters and loglik."""
        n = X.shape[0]
        startprob, transmat, means, variances = params
        prev = -np.inf
        n_done = 0
        for _ in range(self.n_iter):
            log_b = _diag_log_density(X, means, variances)
            gamma, xi_sum, ll = forward_backward(log_b, _log(startprob), _log(transmat))
            startprob, transmat, means, variances = _m_step(
                X, gamma, xi_sum, self.reg_covar
            )
            n_done += 1
            if ll - prev < self.tol * n:
                break
            prev = ll
        log_b = _diag_log_density(X, means, variances)
        ll = forward_backward(log_b, _log(startprob), _log(transmat))[2]
        self._last_n_iter = n_done
        return (startprob, transmat, means, variances), ll

    def _fit_k(self, X, K, rng):
        best = None
        for _ in range(self.n_init):
            seed = int(rng.integers(np.iinfo(np.int32).max))
            params, ll = self._em(X, self._init_params(X, K, seed))
            if best is None or ll > best[1]:
                best = (params, ll, self._last_n_iter)
        return best

    @staticmethod
    def _n_free_params(K, d):
        return (K - 1) + K * (K - 1) + 2 * K * d

    # ---------------------------------------------------------------- interface

    def _fit(self, X, y=None):
        if not _HAS_NUMBA:
            warnings.warn(
                "numba is not installed: GaussianHMMDetector runs its recursions in "
                "pure Python, much slower (pip install tsseg[accelerators])",
                RuntimeWarning,
                stacklevel=2,
            )
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        self._mean = X.mean(axis=0)
        scale = X.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        X = self._prepare(X)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        if self.n_states is not None:
            candidates = [int(self.n_states)]
        else:
            candidates = list(range(1, min(int(self.max_states), n) + 1))

        best = None
        self.bic_ = {}
        for K in candidates:
            params, ll, n_iter = self._fit_k(X, K, rng)
            score = -2.0 * ll + self._n_free_params(K, d) * np.log(n)
            if self.n_states is None:
                self.bic_[K] = float(score)
            if best is None or score < best[0]:
                best = (score, K, params, ll, n_iter)

        _, K, params, ll, n_iter = best
        if self.n_states is None and K == candidates[-1] and K > 1:
            warnings.warn(
                f"GaussianHMMDetector: BIC selected the largest number of states "
                f"tried (max_states={K}); a larger max_states may fit better",
                UserWarning,
                stacklevel=2,
            )
        self.n_states_ = K
        self.startprob_, self.transmat_, self.means_, self.variances_ = params
        self.loglik_ = float(ll)
        self.n_iter_ = int(n_iter)
        return self

    def _predict(self, X):
        X = np.asarray(X, dtype=float)
        d = 1 if X.ndim == 1 else X.shape[1]
        if d != self.means_.shape[1]:
            raise ValueError(
                f"X has {d} channels, but GaussianHMMDetector was fitted on "
                f"{self.means_.shape[1]}"
            )
        X = self._prepare(X)
        log_b = _diag_log_density(X, self.means_, self.variances_)
        path = viterbi(log_b, _log(self.startprob_), _log(self.transmat_))
        # number the states by order of first appearance
        _, first = np.unique(path, return_index=True)
        order = np.argsort(first)
        relabel = np.empty(self.n_states_, dtype=np.int64)
        relabel[np.unique(path)[order]] = np.arange(len(order))
        return relabel[path]
