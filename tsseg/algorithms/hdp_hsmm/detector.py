"""HDP-HSMM state detector: the weak-limit Gibbs sampler of Johnson & Willsky (2013).

The model and the sampler follow Johnson & Willsky (2013), which extends Johnson &
Willsky (2010). The code was checked against the authors' ``pyhsmm`` and against exact
enumeration of the posterior on small problems.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy import stats

from ..base import BaseSegmenter

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
def _forward(cum, logD, logS, logA, logpi, Dmax):
    """Forward messages of an explicit-duration HSMM, log domain.

    cum   (T+1, K)  cumulative emission log-likelihoods, cum[0] = 0
    logD  (K, Dmax) log p(duration = d + 1)
    logS  (K, Dmax) log p(duration >= d + 1)          (right-censored last segment)
    Returns F (T, K): log p(x_0..t, a segment of state k ends at t)
            S (T, K): log p(x_0..t-1, a segment of state k starts at t)
            Fc (K,) : log p(x, the last, censored, segment has state k)
    """
    T = cum.shape[0] - 1
    K = cum.shape[1]
    F = np.full((T, K), -np.inf)
    S = np.full((T, K), -np.inf)
    G = np.empty((K, T))  # S - cum at the start, laid out for the d loop
    Fc = np.full(K, -np.inf)
    for t in range(T):
        if t == 0:
            for k in range(K):
                S[0, k] = logpi[k]
        else:
            for k in range(K):
                m = -np.inf
                for j in range(K):
                    v = F[t - 1, j] + logA[j, k]
                    if v > m:
                        m = v
                if m > -np.inf:
                    acc = 0.0
                    for j in range(K):
                        acc += math.exp(F[t - 1, j] + logA[j, k] - m)
                    S[t, k] = m + math.log(acc)
        for k in range(K):
            G[k, t] = S[t, k] - cum[t, k]
        dm = min(t + 1, Dmax)
        last = t == T - 1
        for k in range(K):
            for c in range(2 if last else 1):
                L = logS if c == 1 else logD
                m = -np.inf
                for d in range(dm):
                    v = G[k, t - d] + L[k, d]
                    if v > m:
                        m = v
                if m > -np.inf:
                    acc = 0.0
                    for d in range(dm):
                        acc += math.exp(G[k, t - d] + L[k, d] - m)
                    if c == 0:
                        F[t, k] = cum[t + 1, k] + m + math.log(acc)
                    else:
                        Fc[k] = cum[t + 1, k] + m + math.log(acc)
    return F, S, Fc


def _sample_log(lp):
    """Index drawn with probabilities proportional to exp(lp), global numpy RNG."""
    p = np.exp(lp - lp.max())
    c = np.cumsum(p)
    return int(np.searchsorted(c, np.random.random() * c[-1], side="right"))


def _block_reduce(X, max_len, features):
    """Block means (and standard deviations) over blocks of ``ceil(T / max_len)`` samples."""
    T = X.shape[0]
    f = int(math.ceil(T / max_len))
    n = int(math.ceil(T / f))
    pad = n * f - T
    Xp = np.vstack([X, np.repeat(X[-1:], pad, axis=0)]) if pad else X
    B = Xp.reshape(n, f, -1)
    # the padded tail block repeats the last sample: its mean is biased by at most pad/f
    R = B.mean(1)
    if features == "meanstd":
        R = np.hstack([R, B.std(1)])
    return R, f


def _znorm(X):
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return (X - X.mean(0)) / sd


class HdpHsmmDetector(BaseSegmenter):
    """Bayesian non-parametric HDP-HSMM state detector (Gibbs sampling).

    Weak-limit HDP-HSMM (Johnson & Willsky, 2013) with Gaussian emissions
    (Normal-Inverse-Wishart prior) and shifted Poisson or negative-binomial
    durations. The number of states is inferred, up to ``n_max_states``. The
    segmentation returned is the last Gibbs sample.

    Parameters
    ----------
    axis : int, default=0
        Axis along which the time index lies.
    alpha : float, default=6.0
        Concentration of the transition distributions around the global weights.
    gamma : float, default=6.0
        Concentration of the global state weights.
    init_state_concentration : float, default=6.0
        Concentration of the initial state distribution.
    n_iter : int, default=20
        Number of Gibbs sweeps.
    n_max_states : int, default=20
        Weak-limit truncation of the number of states.
    trunc : int or None, default=None
        Maximum segment duration, in samples of the input series. ``None`` means no
        truncation.
    kappa0 : float, default=0.25
        NIW pseudo-count on the mean.
    nu0 : float, optional
        NIW degrees of freedom. Defaults to ``d + 1 + emission_strength``.
    prior_mean : float or array-like, default=0.0
        NIW prior mean.
    prior_scale : float or array-like, default=1.0
        Prior expected covariance ``E[Sigma]`` (times the identity if scalar) when ``nu0``
        is None; the inverse-Wishart scale matrix when ``nu0`` is given. With
        ``nu0 = d + 2`` the two readings coincide.
    emission_strength : float, default=100.0
        Pseudo-observations behind ``E[Sigma]`` when ``nu0`` is None: the inverse-Wishart
        prior has ``nu0 = d + 1 + emission_strength`` degrees of freedom and scale
        ``emission_strength * prior_scale``.
    dur_family : {"poisson", "negbin"}, default="poisson"
        Duration distribution, shifted to start at 1.
    dur_alpha : float, default=5.0
        Poisson: Gamma shape of the prior on the rate (strength of the prior); its mean is
        set by ``dur_beta``. Negative binomial: strength of the Beta prior on ``p``.
    dur_beta : float or None, default=None
        Poisson: Gamma rate of the prior on the rate, in samples of the input series.
        ``None`` sets the prior mean duration to a third of the series length, the most
        influential setting of the detector.
    dur_r : float, default=1.0
        Negative binomial: fixed number of failures r; the Beta prior on ``p`` has the same
        mean duration as the Poisson prior.
    max_len : int or None, default=2000
        Series longer than this are reduced by block means over ``ceil(T / max_len)``
        samples before the fit, and the labels are expanded back. ``trunc`` and
        ``dur_beta`` are converted to blocks. ``None`` disables the reduction; a sweep
        then costs O(T^2 K) when ``trunc`` is None.
    block_features : {"mean", "meanstd"}, default="mean"
        Features of a block when the series is reduced: block means, or block means and
        standard deviations.
    normalize : bool, default=True
        z-normalise each channel (after the reduction) before the fit. The emission prior
        is expressed in these units.
    init : {"states", "params"}, default="states"
        Start of the chain: a state sequence drawn from the prior HSMM, as ``pyhsmm``, or
        parameters drawn from the prior, as the previous tsseg detector.

    Attributes
    ----------
    log_likelihood_ : float
        ``log p(X | theta)`` at the last sample, state sequence integrated out, on the
        series actually modelled (reduced and normalised).

    References
    ----------
    Johnson, M. J. and Willsky, A. S. (2013). Bayesian Nonparametric Hidden Semi-Markov
    Models. Journal of Machine Learning Research, 14, 673-701.

    Johnson, M. J. and Willsky, A. S. (2010). The Hierarchical Dirichlet Process Hidden
    Semi-Markov Model. Conference on Uncertainty in Artificial Intelligence (UAI).
    """

    _tags = {
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        "non_deterministic": True,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        axis: int = 0,
        alpha: float = 6.0,
        gamma: float = 6.0,
        init_state_concentration: float = 6.0,
        n_iter: int = 20,
        n_max_states: int = 20,
        trunc: int | None = None,
        *,
        kappa0: float = 0.25,
        nu0: float | None = None,
        prior_mean: Any = 0.0,
        prior_scale: Any = 1.0,
        emission_strength: float = 100.0,
        dur_family: str = "poisson",
        dur_alpha: float = 5.0,
        dur_beta: float | None = None,
        dur_r: float = 1.0,
        max_len: int | None = 2000,
        block_features: str = "mean",
        normalize: bool = True,
        init: str = "states",
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.init_state_concentration = init_state_concentration
        self.n_iter = n_iter
        self.n_max_states = n_max_states
        self.trunc = trunc
        self.kappa0 = kappa0
        self.nu0 = nu0
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.emission_strength = emission_strength
        self.dur_family = dur_family
        self.dur_alpha = dur_alpha
        self.dur_beta = dur_beta
        self.dur_r = dur_r
        self.max_len = max_len
        self.block_features = block_features
        self.normalize = normalize
        self.init = init

        self._states_seq = None
        super().__init__(axis=axis)

    def _prepare_signal(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.axis == 1:
            arr = arr.T
        return arr

    def _check_params(self):
        for name, value, allowed in (
            ("dur_family", self.dur_family, ("poisson", "negbin")),
            ("block_features", self.block_features, ("mean", "meanstd")),
            ("init", self.init, ("states", "params")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if int(self.n_max_states) < 1:
            raise ValueError(f"n_max_states must be >= 1, got {self.n_max_states!r}")
        if int(self.n_iter) < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter!r}")
        for name, value in (
            ("kappa0", self.kappa0),
            ("dur_alpha", self.dur_alpha),
            ("dur_r", self.dur_r),
        ):
            if not float(value) > 0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
        if self.nu0 is None and not float(self.emission_strength) > 0:
            raise ValueError(
                f"emission_strength must be > 0, got {self.emission_strength!r}"
            )
        if self.max_len is not None and int(self.max_len) < 1:
            raise ValueError(f"max_len must be None or >= 1, got {self.max_len!r}")
        if self.trunc is not None and int(self.trunc) < 1:
            raise ValueError(f"trunc must be None or >= 1, got {self.trunc!r}")

    # ------------------------------------------------------------------ parameters

    def _emission_prior(self, d):
        mu0 = (
            np.full(d, float(self.prior_mean))
            if np.isscalar(self.prior_mean)
            else np.asarray(self.prior_mean, float)
        )
        sc = (
            np.eye(d) * float(self.prior_scale)
            if np.isscalar(self.prior_scale)
            else np.asarray(self.prior_scale, float)
        )
        if sc.ndim == 1:
            sc = np.diag(sc)
        if self.nu0 is None:
            nu0 = d + 1.0 + float(self.emission_strength)
            psi0 = float(self.emission_strength) * sc
        else:
            nu0, psi0 = float(self.nu0), sc
        return mu0, float(self.kappa0), nu0, psi0

    @staticmethod
    def _sample_niw(mu0, kappa0, nu0, psi0, n, xbar, scatter):
        if n > 0:
            kappa_n, nu_n = kappa0 + n, nu0 + n
            mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
            dv = xbar - mu0
            psi_n = psi0 + scatter + (kappa0 * n / kappa_n) * np.outer(dv, dv)
        else:
            mu_n, kappa_n, nu_n, psi_n = mu0, kappa0, nu0, psi0
        psi_n = 0.5 * (psi_n + psi_n.T)
        cov = np.atleast_2d(stats.invwishart.rvs(df=nu_n, scale=psi_n))
        mean = np.random.multivariate_normal(mu_n, cov / kappa_n)
        return mean, cov

    @staticmethod
    def _gauss_loglik(X, mean, cov):
        d = X.shape[1]
        c = cov + 1e-10 * np.trace(cov) / d * np.eye(d)
        L = np.linalg.cholesky(c)
        sol = np.linalg.solve(L, (X - mean).T)
        return (
            -0.5 * (sol * sol).sum(0)
            - np.log(np.diag(L)).sum()
            - 0.5 * d * np.log(2 * np.pi)
        )

    def _dur_logs(self, lam_or_p, Dmax):
        """(logD, logS), each (K, Dmax): log pmf and log survival of d = 1..Dmax."""
        k = np.arange(Dmax, dtype=float)  # d - 1
        if self.dur_family == "poisson":
            lam = lam_or_p[:, None]
            logD = stats.poisson.logpmf(k[None, :], lam)
            logS = stats.poisson.logsf(k[None, :] - 1, lam)
        else:
            q = 1.0 - lam_or_p[:, None]  # scipy success probability
            logD = stats.nbinom.logpmf(k[None, :], self.dur_r, q)
            logS = stats.nbinom.logsf(k[None, :] - 1, self.dur_r, q)
        return np.ascontiguousarray(logD), np.ascontiguousarray(logS)

    def _dur_sample(self, durs, cens, a0, b0, par):
        """One duration parameter from its posterior; ``cens`` are censored lengths (>= d),
        imputed under the current parameter ``par`` of the state."""
        durs = list(durs)
        for c in cens:
            durs.append(self._rvs_ge(c, par))
        x = np.asarray(durs, float) - 1.0
        if self.dur_family == "poisson":
            return max(np.random.gamma(a0 + x.sum(), 1.0 / (b0 + len(x))), 1e-3)
        return np.random.beta(a0 + x.sum(), b0 + len(x) * self.dur_r)

    def _rvs_ge(self, c, par):
        """Duration >= c under the duration parameter ``par``."""
        if self.dur_family == "poisson":
            mean, sd = par + 1, math.sqrt(par + 1)
        else:
            q = 1.0 - par
            mean = 1 + self.dur_r * par / q
            sd = math.sqrt(self.dur_r * par) / q
        hi = int(max(c, mean) + 50 * sd + 50)
        d = np.arange(c, hi + 1)
        if self.dur_family == "poisson":
            lp = stats.poisson.logpmf(d - 1, par)
        else:
            lp = stats.nbinom.logpmf(d - 1, self.dur_r, 1.0 - par)
        if not np.isfinite(lp.max()):
            return int(c)
        return int(d[_sample_log(lp)])

    # ------------------------------------------------------------------ fit

    def _fit(self, X, y=None):
        self._check_params()
        if not _HAS_NUMBA:
            warnings.warn(
                "numba is not installed: HdpHsmmDetector runs its messages in pure "
                "Python, much slower (pip install tsseg[accelerators])",
                RuntimeWarning,
                stacklevel=2,
            )
        X = self._prepare_signal(X)
        T0 = X.shape[0]
        if T0 == 0:
            raise ValueError("HdpHsmmDetector needs a non-empty series")
        self._factor = 1
        if self.max_len is not None and T0 > self.max_len:
            X, self._factor = _block_reduce(X, int(self.max_len), self.block_features)
        if self.normalize:
            X = _znorm(X)
        z = self._gibbs(X, self._factor)
        self._states_seq = np.repeat(z, self._factor)[:T0]
        return self

    def _gibbs(self, X, factor=1):
        T, d = X.shape
        K = self.n_max_states
        # no truncation by default; trunc is in samples of the input series
        Dmax = T if self.trunc is None else int(min(T, math.ceil(self.trunc / factor)))
        mu0, kappa0, nu0, psi0 = self._emission_prior(d)

        # Prior mean duration m = T/3, the most influential setting. A Poisson duration
        # has a standard deviation of sqrt(m): m is not a bound, but it sets the scale of
        # the segmentation.
        m = max(2.0, T / 3.0)
        if self.dur_family == "poisson":
            a0 = float(self.dur_alpha)
            b0 = float(self.dur_beta) * factor if self.dur_beta is not None else a0 / m
        else:
            b0 = float(self.dur_alpha) + 2.0  # E[p/(1-p)] = a0 / (b0 - 1)
            a0 = (b0 - 1.0) * (m - 1.0) / float(self.dur_r)

        # initial parameters from the prior
        emis = [
            self._sample_niw(mu0, kappa0, nu0, psi0, 0, None, None) for _ in range(K)
        ]
        if self.dur_family == "poisson":
            dpar = np.maximum(np.random.gamma(a0, 1.0 / b0, size=K), 1e-3)
        else:
            dpar = np.random.beta(a0, b0, size=K)
        beta = np.random.dirichlet(np.full(K, self.gamma / K))
        beta = np.maximum(beta, 1e-300)
        beta /= beta.sum()
        full = np.array([self._dirichlet(self.alpha * beta) for _ in range(K)])
        A = self._hsmm_rows(full, beta)
        pi0 = self._dirichlet(np.full(K, self.init_state_concentration / K))

        # start from a state sequence drawn from the prior, as pyhsmm;
        # init="params" (the previous tsseg start) ends in too fine segmentations
        z = segs = None
        if self.init == "states":
            z, segs = self._generate(T, pi0, A, dpar)
        for _ in range(self.n_iter):
            if z is not None:
                emis, dpar, beta, full, A, pi0 = self._resample_params(
                    X,
                    z,
                    segs,
                    emis,
                    dpar,
                    beta,
                    full,
                    A,
                    pi0,
                    mu0,
                    kappa0,
                    nu0,
                    psi0,
                    a0,
                    b0,
                )
            # --- states
            logB = np.column_stack([self._gauss_loglik(X, *emis[k]) for k in range(K)])
            cum = np.vstack([np.zeros((1, K)), np.cumsum(logB, axis=0)])
            logD, logS = self._dur_logs(dpar, Dmax)
            with np.errstate(divide="ignore"):
                logA, logpi = np.log(A), np.log(pi0)
            F, S, Fc = _forward(cum, logD, logS, logA, logpi, Dmax)
            if not np.isfinite(Fc).any():
                raise ValueError(
                    "no segmentation of the series has a positive probability "
                    "under the model: increase trunc or n_max_states"
                )
            self.log_likelihood_ = float(np.logaddexp.reduce(Fc))
            z, segs = self._backward(F, S, Fc, cum, logD, logS, logA, Dmax)

        return z

    def _resample_params(
        self, X, z, segs, emis, dpar, beta, full, A, pi0, mu0, kappa0, nu0, psi0, a0, b0
    ):
        T = X.shape[0]
        K = self.n_max_states
        # --- emissions
        for k in range(K):
            Xk = X[z == k]
            n = len(Xk)
            if n:
                xb = Xk.mean(0)
                C = Xk - xb
                emis[k] = self._sample_niw(mu0, kappa0, nu0, psi0, n, xb, C.T @ C)
            else:
                emis[k] = self._sample_niw(mu0, kappa0, nu0, psi0, 0, None, None)

        # --- durations (the last segment is right-censored)
        for k in range(K):
            full_d = [L for (s, L) in segs[:-1] if s == k]
            cens = [segs[-1][1]] if segs[-1][0] == k else []
            dpar[k] = self._dur_sample(full_d, cens, a0, b0, dpar[k])

        # --- transitions, global weights
        n = np.zeros((K, K))
        for (u, _), (v, _) in zip(segs[:-1], segs[1:], strict=True):
            n[u, v] += 1
        beta, full, A = self._resample_transitions(n, full, beta, T)
        first = np.zeros(K)
        first[segs[0][0]] = 1
        pi0 = self._dirichlet(self.init_state_concentration / K + first)

        return emis, dpar, beta, full, A, pi0

    def _generate(self, T, pi0, A, dpar):
        """State sequence drawn from the prior HSMM (pyhsmm ``generate_states``)."""
        z = np.empty(T, dtype=np.int64)
        segs, t, p = [], 0, pi0
        while t < T:
            if not p.sum() > 0:
                # no other state to go to (n_max_states = 1): the segment runs to the end
                k, L = segs[-1]
                z[t:] = k
                segs[-1] = (k, L + T - t)
                break
            k = int(
                np.searchsorted(
                    np.cumsum(p), np.random.random() * p.sum(), side="right"
                )
            )
            if self.dur_family == "poisson":
                L = 1 + np.random.poisson(dpar[k])
            else:
                L = 1 + np.random.negative_binomial(self.dur_r, 1.0 - dpar[k])
            L = int(min(L, T - t))
            z[t : t + L] = k
            segs.append((k, L))
            t += L
            p = A[k]
        return z, segs

    def _resample_transitions(self, n, full, beta, T):
        """Weak-limit HDP transitions of an HSMM (Johnson & Willsky 2013, eqs. 46-49).

        ``n`` counts the transitions between consecutive segments (zero diagonal),
        ``full`` holds the rows pi_j ~ Dir(alpha beta) including their self-transition
        mass. Each exit from j is preceded by a geometric number of virtual
        self-transitions, support {0, 1, ...}, success 1 - pi_jj: their sum is one negative
        binomial draw. Capped: success >= 1/(T+1), sum <= T.
        """
        K = n.shape[0]
        aug = n.copy()
        exits = n.sum(1)
        for j in range(K):
            if exits[j] > 0:
                p = max(1.0 - full[j, j], 1.0 / (T + 1))
                aug[j, j] = min(np.random.negative_binomial(int(exits[j]), p), T)
        mtab = np.zeros((K, K))
        for j in range(K):
            for k in range(K):
                c = int(aug[j, k])
                if c:
                    conc = self.alpha * beta[k]
                    mtab[j, k] = (
                        np.random.random(c) < conc / (conc + np.arange(c))
                    ).sum()
        beta = self._dirichlet(self.gamma / K + mtab.sum(0))
        full = np.array([self._dirichlet(self.alpha * beta + aug[j]) for j in range(K)])
        return beta, full, self._hsmm_rows(full, beta)

    @staticmethod
    def _dirichlet(a):
        g = np.random.gamma(np.maximum(a, 1e-10))
        s = g.sum()
        if not s > 0:
            g = np.maximum(a, 1e-10)
            s = g.sum()
        return g / s

    @staticmethod
    def _hsmm_rows(full, beta):
        """HSMM transition rows: diagonal removed, renormalised; a row whose off-diagonal
        mass underflows falls back to the global weights of the other states."""
        A = full.copy()
        np.fill_diagonal(A, 0.0)
        s = A.sum(1)
        for j in np.flatnonzero(~(s > 1e-300)):
            A[j] = beta
            A[j, j] = 0.0
            s[j] = A[j].sum()
        s[~(s > 0)] = 1.0  # no other state to go to (K = 1): the row stays zero
        return A / s[:, None]

    @staticmethod
    def _backward(F, S, Fc, cum, logD, logS, logA, Dmax):
        T, K = F.shape
        z = np.empty(T, dtype=np.int64)
        segs = []
        k = _sample_log(Fc)
        t, censored = T, True
        while t > 0:
            dm = min(t, Dmax)
            dd = np.arange(1, dm + 1)
            s = t - dd
            Ld = (logS if censored else logD)[k, :dm]
            lp = S[s, k] + Ld + cum[t, k] - cum[s, k]
            L = int(dd[_sample_log(lp)])
            z[t - L : t] = k
            segs.append((k, L))
            t -= L
            censored = False
            if t > 0:
                k = _sample_log(F[t - 1] + logA[:, k])
        segs.reverse()
        return z, segs

    def _predict(self, X):
        if self._states_seq is None:
            return np.zeros(len(X), dtype=int)
        return self._states_seq

    def get_fitted_params(self):
        return {}
