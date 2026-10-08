"""Change-point score of Liu et al. (2013) by relative density-ratio estimation.

Written for tsseg from the paper only:

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
        Row ``t`` is ``[y(t), y(t + 1), ..., y(t + k - 1)]`` flattened, so a
        multivariate subsequence concatenates all channels into one vector
        (footnote 1 of the paper; the order of the coordinates does not
        matter for the Gaussian kernel).
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


def _fold_masks(n_samples: int, n_folds: int) -> np.ndarray:
    """Boolean test masks of shape (n_folds, n_samples), sample ``i`` in fold ``i % n_folds``."""
    return (np.arange(n_samples)[None, :] % n_folds) == np.arange(n_folds)[:, None]


def relative_pe_divergence(
    K_num: np.ndarray,
    K_den: np.ndarray,
    alpha: float,
    lambdas: np.ndarray,
    n_folds: int,
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
    K_den : np.ndarray of shape (n_kernels, n', n)
        ``K_den[s, j, l] = K_s(Y'_j, Y_l)``, between the samples of
        :math:`P'` and the centres.
    alpha : float
        Mixture weight, ``0 <= alpha < 1``.
    lambdas : np.ndarray of shape (n_lambdas,)
        Candidate regularisation parameters (non-negative).
    n_folds : int
        Number of cross-validation folds (at least 2). Samples are assigned
        to folds by index modulo ``n_folds``. When there is a single
        candidate pair, no cross-validation is run.

    Returns
    -------
    divergence : float
        Estimate of :math:`\mathrm{PE}_\alpha(P \| P')`.
    kernel_index : int
        Index of the selected kernel width.
    lambda_index : int
        Index of the selected regularisation parameter.
    """
    n_kernels, n_num, n_centres = K_num.shape
    n_den = K_den.shape[1]
    lambdas = np.asarray(lambdas, dtype=float)
    eye = np.eye(n_centres)

    if n_kernels * lambdas.size == 1:
        best_s, best_l = 0, 0
    else:
        test_num = _fold_masks(n_num, n_folds)
        test_den = _fold_masks(n_den, n_folds)
        ridge = lambdas[None, :, None, None] * eye
        loss = np.zeros((n_kernels, lambdas.size))
        for f in range(n_folds):
            A = K_num[:, ~test_num[f], :]
            B = K_den[:, ~test_den[f], :]
            H = alpha / A.shape[1] * (A.transpose(0, 2, 1) @ A)
            H += (1.0 - alpha) / B.shape[1] * (B.transpose(0, 2, 1) @ B)
            h = A.mean(axis=1)
            # one system per (kernel width, lambda): theta has shape (s, q, l, 1)
            rhs = np.broadcast_to(
                h[:, None, :, None], (n_kernels, lambdas.size, n_centres, 1)
            )
            theta = np.linalg.solve(H[:, None, :, :] + ridge, rhs)
            g_num = (K_num[:, None, test_num[f], :] @ theta)[..., 0]
            g_den = (K_den[:, None, test_den[f], :] @ theta)[..., 0]
            loss += (
                0.5 * alpha * np.mean(g_num**2, axis=2)
                + 0.5 * (1.0 - alpha) * np.mean(g_den**2, axis=2)
                - np.mean(g_num, axis=2)
            )
        best_s, best_l = np.unravel_index(int(np.argmin(loss)), loss.shape)

    A = K_num[best_s]
    B = K_den[best_s]
    H = alpha * (A.T @ A) / n_num + (1.0 - alpha) * (B.T @ B) / n_den
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
    own cross-validated kernel width and regularisation. The kernel widths
    are ``sigma_factors`` times the median pairwise distance between the
    ``2n`` subsequences of the two sets (Section 4.1).

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
        Candidate kernel widths, as multiples of the median distance.
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
    offset = n + (k - 1) // 2
    iu = np.triu_indices(2 * n, k=1)
    # The linear systems are tiny (n x n): BLAS threads only add overhead.
    limit = threadpool_limits(1) if threadpool_limits is not None else nullcontext()
    with limit:
        for t in range(n_starts):
            U = S[t : t + 2 * n]
            G = _squared_distances(U)
            d_med = float(np.median(np.sqrt(G[iu])))
            if d_med <= 0.0:
                # All subsequences identical up to ties: no evidence of change.
                scores[t + offset] = 0.0
                continue
            sigmas = factors * d_med
            K = np.exp(-G[None, :, :] / (2.0 * sigmas[:, None, None] ** 2))
            first, second = slice(0, n), slice(n, 2 * n)
            forward, _, _ = relative_pe_divergence(
                K[:, first, first], K[:, second, first], alpha, lam, n_folds
            )
            backward, _, _ = relative_pe_divergence(
                K[:, second, second], K[:, first, second], alpha, lam, n_folds
            )
            scores[t + offset] = forward + backward
    return scores
