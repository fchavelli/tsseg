"""Change-point score of Liu et al. (2013) by relative density-ratio estimation.

Port of the authors' MATLAB code (``change_detection.m``, ``RelULSIF.m`` and
``lib/``, https://github.com/anewgithubname/change_detection, commit
``496042e``, MIT licence, see ``LICENSE`` in this directory) for the paper

    S. Liu, M. Yamada, N. Collier, M. Sugiyama, "Change-point detection in
    time-series data by relative density-ratio estimation", Neural Networks
    43:72-83, 2013 (arXiv:1203.0453).

Notation follows the paper: ``k`` is the subsequence length, ``n`` the number
of subsequences in each of the two compared sets, ``alpha`` the mixture
weight of the relative density ratio.
"""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

try:  # installed with scikit-learn
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

__all__ = [
    "subsequences",
    "cv_folds",
    "split_kernels",
    "relative_pe_divergence",
    "rulsif_scores",
]


def subsequences(X: np.ndarray, k: int) -> np.ndarray:
    """Stack the subsequences ``Y(t)`` of length ``k`` (Section 2).

    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints, n_channels)
        Time series, time along the first axis.
    k : int
        Subsequence length.

    Returns
    -------
    np.ndarray of shape (n_timepoints - k + 1, k * n_channels)
        Row ``t`` holds ``y(t), ..., y(t + k - 1)``; a multivariate
        subsequence concatenates all channels into one vector (footnote 1 of
        the paper; the order of the coordinates does not matter for the
        Gaussian kernel).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    windows = np.lib.stride_tricks.sliding_window_view(X, k, axis=0)
    return windows.reshape(windows.shape[0], -1)


def _squared_distances(U: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances between the rows of ``U``."""
    sq = np.einsum("ij,ij->i", U, U)
    G = sq[:, None] + sq[None, :] - 2.0 * (U @ U.T)
    np.maximum(G, 0.0, out=G)
    np.fill_diagonal(G, 0.0)
    return G


def cv_folds(
    n_samples: int, n_folds: int, n_candidates: int, seed: int = 0
) -> np.ndarray:
    """Cross-validation folds of ``RelULSIF.m``, one partition per candidate.

    The official code resets the random generator (``rng default``) at every
    estimate, draws one permutation for the kernel centres, then, for each
    (kernel width, regularisation) candidate in turn, one permutation of the
    numerator samples and one of the denominator samples; the sample at
    position ``p`` of a permutation goes to fold ``floor(p * n_folds /
    n_samples)``. The folds are therefore random but the same for every
    estimate. MATLAB's generator cannot be reproduced, so NumPy's is used
    with a fixed seed.

    Parameters
    ----------
    n_samples : int
        Number of samples in each of the two sets.
    n_folds : int
        Number of folds.
    n_candidates : int
        Number of (kernel width, regularisation) candidates.
    seed : int, default=0
        Seed of the generator.

    Returns
    -------
    np.ndarray of shape (2, n_candidates, n_samples)
        Fold of every sample, for the numerator set (``[0]``) and the
        denominator set (``[1]``), for every candidate.
    """
    rng = np.random.default_rng(seed)
    rng.permutation(n_samples)  # kernel centres: all samples when n <= 100
    split = np.arange(n_samples) * n_folds // n_samples
    folds = np.empty((2, n_candidates, n_samples), dtype=int)
    for c in range(n_candidates):
        for side in range(2):
            folds[side, c, rng.permutation(n_samples)] = split
    return folds


def relative_pe_divergence(
    K_num: np.ndarray,
    K_den: np.ndarray,
    alpha: float,
    lambdas: np.ndarray,
    n_folds: int,
    folds: np.ndarray | None = None,
) -> tuple[float, int, int]:
    r"""Estimate :math:`\mathrm{PE}_\alpha(P \| P')` by RuLSIF (Section 3.4).

    The relative density ratio :math:`r_\alpha = p / (\alpha p + (1 - \alpha)
    p')` is modelled as :math:`g(Y) = \sum_\ell \theta_\ell K(Y, Y_\ell)`,
    the centres :math:`Y_\ell` being the samples of :math:`P`. For each
    candidate kernel (first axis of ``K_num`` and ``K_den``) and each
    regularisation :math:`\lambda`, :math:`\theta = (\hat H + \lambda
    I)^{-1} \hat h` (equation 8 with the :math:`\hat H` of Section 3.4.2) is
    fitted on ``n_folds - 1`` folds and scored on the held-out fold by the
    empirical squared loss

    .. math::

        J = \frac{\alpha}{2} \overline{g(Y_i)^2}
            + \frac{1 - \alpha}{2} \overline{g(Y'_j)^2}
            - \overline{g(Y_i)}.

    The pair with the lowest mean held-out loss is refitted on all samples,
    which gives

    .. math::

        \widehat{\mathrm{PE}}_\alpha = -\frac{\alpha}{2n} \sum_i g(Y_i)^2
            - \frac{1 - \alpha}{2n} \sum_j g(Y'_j)^2
            + \frac{1}{n} \sum_i g(Y_i) - \frac{1}{2}.

    Parameters
    ----------
    K_num : np.ndarray of shape (n_kernels, n, n)
        ``K_num[s, i, l] = K_s(Y_i, Y_l)``, kernel values between the samples
        of :math:`P` and the centres, for each candidate kernel width.
    K_den : np.ndarray of shape (n_kernels, n, n)
        ``K_den[s, j, l] = K_s(Y'_j, Y_l)``, between the samples of
        :math:`P'` and the centres. Both sets have the same size.
    alpha : float
        Mixture weight, ``0 <= alpha < 1``.
    lambdas : np.ndarray of shape (n_lambdas,)
        Candidate regularisation parameters (non-negative).
    n_folds : int
        Number of cross-validation folds (at least 2). When there is a single
        candidate pair, no cross-validation is run.
    folds : np.ndarray of shape (2, n_kernels * n_lambdas, n), optional
        Fold of every sample per candidate, candidates ordered kernel width
        first (see :func:`cv_folds`, the default).

    Returns
    -------
    divergence : float
        Estimate of :math:`\mathrm{PE}_\alpha(P \| P')`.
    kernel_index : int
        Index of the selected kernel width.
    lambda_index : int
        Index of the selected regularisation parameter.
    """
    n_kernels, n_samples, n_centres = K_num.shape
    lambdas = np.asarray(lambdas, dtype=float)
    n_lambdas = lambdas.size
    eye = np.eye(n_centres)

    if n_kernels * n_lambdas == 1:
        best_s, best_l = 0, 0
    else:
        n_cand = n_kernels * n_lambdas
        if folds is None:
            folds = cv_folds(n_samples, n_folds, n_cand)
        # candidate c = (kernel width c // n_lambdas, lambda c % n_lambdas)
        kernel_of = np.repeat(np.arange(n_kernels), n_lambdas)
        Kn = K_num[kernel_of]
        Kd = K_den[kernel_of]
        # Gram matrices and sums over all samples; each fold subtracts its
        # held-out rows, which is cheaper than gathering the training rows.
        gram_n = (K_num.transpose(0, 2, 1) @ K_num)[kernel_of]
        gram_d = (K_den.transpose(0, 2, 1) @ K_den)[kernel_of]
        sum_n = K_num.sum(axis=1)[kernel_of]
        ridge = np.tile(lambdas, n_kernels)[:, None, None] * eye
        loss = np.zeros(n_cand)
        for f in range(n_folds):
            # every candidate holds out the same number of samples in fold f
            te_n = np.nonzero(folds[0] == f)[1].reshape(n_cand, -1, 1)
            te_d = np.nonzero(folds[1] == f)[1].reshape(n_cand, -1, 1)
            A = np.take_along_axis(Kn, te_n, axis=1)
            B = np.take_along_axis(Kd, te_d, axis=1)
            n_tr = n_samples - A.shape[1]
            m_tr = n_samples - B.shape[1]
            H = alpha / n_tr * (gram_n - A.transpose(0, 2, 1) @ A)
            H += (1.0 - alpha) / m_tr * (gram_d - B.transpose(0, 2, 1) @ B)
            h = ((sum_n - A.sum(axis=1)) / n_tr)[..., None]
            theta = np.linalg.solve(H + ridge, h)
            g_num = (A @ theta)[..., 0]
            g_den = (B @ theta)[..., 0]
            loss += (
                0.5 * alpha * np.mean(g_num**2, axis=1)
                + 0.5 * (1.0 - alpha) * np.mean(g_den**2, axis=1)
                - np.mean(g_num, axis=1)
            )
        best_s, best_l = divmod(int(np.argmin(loss)), n_lambdas)

    A = K_num[best_s]
    B = K_den[best_s]
    H = alpha * (A.T @ A) / A.shape[0] + (1.0 - alpha) * (B.T @ B) / B.shape[0]
    h = A.mean(axis=0)
    theta = np.linalg.solve(H + lambdas[best_l] * eye, h)
    g_num = A @ theta
    g_den = B @ theta
    divergence = (
        -0.5 * alpha * np.mean(g_num**2)
        - 0.5 * (1.0 - alpha) * np.mean(g_den**2)
        + np.mean(g_num)
        - 0.5
    )
    return float(divergence), int(best_s), int(best_l)


def split_kernels(U: np.ndarray, sigma_factors: np.ndarray) -> np.ndarray | None:
    r"""Gaussian kernel matrices between the ``2n`` samples of one split.

    As ``change_detection.m`` and ``RelULSIF.m``: every coordinate is divided
    by its standard deviation over the ``2n`` samples (``ddof=1``, no
    centring; constant coordinates are left as they are), then the kernel
    widths are ``sigma_factors`` times :math:`\sqrt{\operatorname{median}(d^2)
    / 2}`, the median running over the non-zero squared pairwise distances
    (``lib/comp_med.m``), and :math:`K(Y, Y') = \exp(-\lVert Y - Y'
    \rVert^2 / (2 \sigma^2))` (``lib/kernel_gau.m``).

    Parameters
    ----------
    U : np.ndarray of shape (2n, dim)
        The two sets of subsequences, one per row, the first set on top.
    sigma_factors : np.ndarray of shape (n_kernels,)
        Candidate kernel widths, as multiples of the median scale.

    Returns
    -------
    np.ndarray of shape (n_kernels, 2n, 2n) or None
        Kernel matrices, ``None`` when all samples are identical.
    """
    std = U.std(axis=0, ddof=1)
    U = U / np.where(std > 0.0, std, 1.0)
    G = _squared_distances(U)
    d2 = G[np.triu_indices(G.shape[0], k=1)]
    d2 = d2[d2 > 0.0]
    if d2.size == 0:
        return None
    sigmas = np.asarray(sigma_factors, dtype=float) * np.sqrt(0.5 * np.median(d2))
    return np.exp(-G[None, :, :] / (2.0 * sigmas[:, None, None] ** 2))


def rulsif_scores(
    X: np.ndarray,
    k: int = 10,
    n: int = 50,
    alpha: float = 0.1,
    sigma_factors: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
    lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
    n_folds: int = 5,
) -> np.ndarray:
    r"""Symmetric RuLSIF change-point score of every admissible split.

    For each start ``t``, the sets :math:`\mathcal{Y}(t) = \{Y(t), \dots,
    Y(t + n - 1)\}` and :math:`\mathcal{Y}(t + n)` are compared by
    :math:`\mathrm{PE}_\alpha(P_t \| P_{t+n}) + \mathrm{PE}_\alpha(P_{t+n} \|
    P_t)` (Sections 3.1 and 3.4.1), each term estimated separately with its
    own cross-validated kernel width and regularisation, as the official
    code does (``change_detection.m`` run forward and on the reversed
    series, then summed, ``demo.m``). As in that code, every coordinate of
    the ``2n`` subsequences is first divided by its standard deviation over
    them, and the kernel widths are ``sigma_factors`` times
    :math:`\sqrt{\operatorname{median}(d^2) / 2}`, :math:`d` ranging over
    the non-zero pairwise distances between the ``2n`` subsequences
    (``lib/comp_med.m``).

    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints,) or (n_timepoints, n_channels)
        Time series, time along the first axis.
    k : int, default=10
        Subsequence length.
    n : int, default=50
        Number of subsequences in each set.
    alpha : float, default=0.1
        Mixture weight of the relative density ratio.
    sigma_factors : tuple of float, default=(0.6, 0.8, 1.0, 1.2, 1.4)
        Candidate kernel widths, as multiples of the median scale above.
    lambdas : tuple of float, default=(1e-3, 1e-2, 1e-1, 1, 10)
        Candidate regularisation parameters.
    n_folds : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        Score aligned on the time axis; ``nan`` where it is undefined (the
        first and last ``n + (k - 1) // 2`` points, roughly). The score of
        start ``t`` is placed at ``t + n + (k - 1) // 2``: the point where
        the subsequences overlapping the boundary between the two sets are
        split evenly between them.
    """
    X = np.asarray(X, dtype=float)
    n_timepoints = X.shape[0]
    S = subsequences(X, k)
    n_starts = S.shape[0] - 2 * n + 1
    scores = np.full(n_timepoints, np.nan)
    if n_starts < 1:
        return scores

    factors = np.asarray(sigma_factors, dtype=float)
    lam = np.asarray(lambdas, dtype=float)
    folds = cv_folds(n, n_folds, factors.size * lam.size)
    offset = n + (k - 1) // 2
    first, second = slice(0, n), slice(n, 2 * n)
    # The linear systems are tiny (n x n): BLAS threads only add overhead.
    limit = threadpool_limits(1) if threadpool_limits is not None else nullcontext()
    with limit:
        for t in range(n_starts):
            K = split_kernels(S[t : t + 2 * n], factors)
            if K is None:
                # All subsequences identical: no evidence of change.
                scores[t + offset] = 0.0
                continue
            forward, _, _ = relative_pe_divergence(
                K[:, first, first], K[:, second, first], alpha, lam, n_folds, folds
            )
            backward, _, _ = relative_pe_divergence(
                K[:, second, second], K[:, first, second], alpha, lam, n_folds, folds
            )
            scores[t + offset] = forward + backward
    return scores
