"""Log-space forward-backward, Baum-Welch sufficient statistics and Viterbi.

Standard algorithms for a discrete-state HMM (Rabiner, 1989), in log space so that
transitions estimated at exactly zero cannot cause a division by zero.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # numba is optional (tsseg[accelerators]): same code, interpreted
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

else:
    _HAS_NUMBA = True


@njit(cache=True)
def _logsumexp(a):
    m = a.max()
    if m == -np.inf:
        return -np.inf
    s = 0.0
    for v in a:
        s += np.exp(v - m)
    return m + np.log(s)


@njit(cache=True)
def forward_backward(log_b, log_pi, log_a):
    """Posterior marginals and expected transition counts.

    Parameters
    ----------
    log_b : ndarray of shape (n, K)
        Log emission densities ``log p(x_t | s_t = k)``.
    log_pi : ndarray of shape (K,)
        Log initial probabilities.
    log_a : ndarray of shape (K, K)
        Log transition matrix, ``log_a[i, j] = log p(s_t = j | s_{t-1} = i)``.

    Returns
    -------
    gamma : ndarray of shape (n, K)
        ``p(s_t = k | x_{1:n})``.
    xi_sum : ndarray of shape (K, K)
        ``sum_t p(s_t = i, s_{t+1} = j | x_{1:n})``.
    loglik : float
        ``log p(x_{1:n})``.
    """
    n, K = log_b.shape
    la = np.empty((n, K))
    lb = np.empty((n, K))
    tmp = np.empty(K)
    for k in range(K):
        la[0, k] = log_pi[k] + log_b[0, k]
    for t in range(1, n):
        for j in range(K):
            for i in range(K):
                tmp[i] = la[t - 1, i] + log_a[i, j]
            la[t, j] = _logsumexp(tmp) + log_b[t, j]
    loglik = _logsumexp(la[n - 1])
    for k in range(K):
        lb[n - 1, k] = 0.0
    for t in range(n - 2, -1, -1):
        for i in range(K):
            for j in range(K):
                tmp[j] = log_a[i, j] + log_b[t + 1, j] + lb[t + 1, j]
            lb[t, i] = _logsumexp(tmp)
    gamma = np.empty((n, K))
    for t in range(n):
        for k in range(K):
            gamma[t, k] = np.exp(la[t, k] + lb[t, k] - loglik)
    xi_sum = np.zeros((K, K))
    for t in range(n - 1):
        for i in range(K):
            for j in range(K):
                xi_sum[i, j] += np.exp(
                    la[t, i] + log_a[i, j] + log_b[t + 1, j] + lb[t + 1, j] - loglik
                )
    return gamma, xi_sum, loglik


@njit(cache=True)
def viterbi(log_b, log_pi, log_a):
    """Most probable state sequence (ties go to the lowest state index)."""
    n, K = log_b.shape
    delta = np.empty(K)
    new = np.empty(K)
    psi = np.zeros((n, K), dtype=np.int64)
    for k in range(K):
        delta[k] = log_pi[k] + log_b[0, k]
    for t in range(1, n):
        for j in range(K):
            best = -np.inf
            arg = 0
            for i in range(K):
                v = delta[i] + log_a[i, j]
                if v > best:
                    best = v
                    arg = i
            new[j] = best + log_b[t, j]
            psi[t, j] = arg
        for k in range(K):
            delta[k] = new[k]
    path = np.empty(n, dtype=np.int64)
    best = -np.inf
    arg = 0
    for k in range(K):
        if delta[k] > best:
            best = delta[k]
            arg = k
    path[n - 1] = arg
    for t in range(n - 1, 0, -1):
        path[t - 1] = psi[t, path[t]]
    return path
