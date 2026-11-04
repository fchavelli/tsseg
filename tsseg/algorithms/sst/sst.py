"""Singular spectrum transformation scores, exact and FFT-accelerated rSVD.

Reimplemented from the papers (Ide & Tsuda, SDM 2007; Weber & Lenz, IEEE
Access 2025), without third-party code.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, next_fast_len, rfft

__all__ = ["sst_scores"]

# Elements of the largest work array of one batch (about 64 MB of complex).
_BATCH_ELEMENTS = 4_000_000


def _preprocess(X: np.ndarray) -> np.ndarray:
    """Unit variance and mean 3 for every channel (Ide & Tsuda 2007, Sec. 5.1)."""
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - X.mean(axis=0)) / std + 3.0


def _hankel_windows(X: np.ndarray, N: int, ends: np.ndarray) -> np.ndarray:
    """Generating vectors of the Hankel matrices ending before ``ends``.

    Returns an array of shape ``(len(ends), n_channels, 2N - 1)``: for the
    position ``t``, channel ``c`` holds ``X[t - 2N + 1 : t, c]``, whose
    Hankel matrix ``H[i, m] = h[i + m]`` (``N x N``) has the lagged vectors
    ``s(t - N), ..., s(t - 1)`` as columns.
    """
    windows = sliding_window_view(X, 2 * N - 1, axis=0)  # (T - 2N + 2, d, 2N - 1)
    return windows[ends - (2 * N - 1)]


def _left_vectors_exact(h: np.ndarray, N: int, rank: int) -> np.ndarray:
    """``rank`` leading left singular vectors of ``[H_1, ..., H_d]`` (full SVD)."""
    B, d, _ = h.shape
    H = sliding_window_view(h, N, axis=-1)  # (B, d, N, N), H[b, c, m, i] = h[m + i]
    H = np.transpose(H, (0, 2, 1, 3)).reshape(B, N, d * N)
    U = np.linalg.svd(H, full_matrices=False, compute_uv=True)[0]
    return U[:, :, :rank]


def _hankel_matmul(h_fft: np.ndarray, A: np.ndarray, N: int, nfft: int) -> np.ndarray:
    """Products ``H_c A_c`` of square Hankel matrices given by their FFT.

    ``h_fft`` has shape ``(B, d, nfft // 2 + 1)`` and ``A`` shape
    ``(B, d, N, G)``; returns ``(B, d, N, G)``. The product is a
    cross-correlation: the columns are flipped, convolved by FFT and the
    entries ``N - 1`` to ``2N - 2`` kept (Weber & Lenz 2025, Alg. 4).
    """
    A_fft = rfft(A[:, :, ::-1, :], n=nfft, axis=2)
    return irfft(A_fft * h_fft[:, :, :, None], n=nfft, axis=2)[:, :, N - 1 : 2 * N - 1]


def _left_vectors_rsvd(
    h: np.ndarray, N: int, rank: int, omega: np.ndarray, n_iter: int
) -> np.ndarray:
    """``rank`` leading left singular vectors of ``[H_1, ..., H_d]`` by rSVD.

    Randomised SVD with subspace iterations (Halko, Martinsson & Tropp 2011,
    Alg. 4.4), every Hankel product computed by FFT (Weber & Lenz 2025,
    Alg. 3-5). Each block ``H_c`` is square and Hankel, hence symmetric, so
    ``[H_1, ..., H_d]^T y`` stacks the products ``H_c y``.
    """
    B, d, _ = h.shape
    nfft = next_fast_len(2 * N - 1, real=True)
    h_fft = rfft(h, n=nfft, axis=2)
    G = omega.shape[1]

    def times_h(A: np.ndarray) -> np.ndarray:  # (B, d*N, G) -> (B, N, G)
        return _hankel_matmul(h_fft, A.reshape(B, d, N, -1), N, nfft).sum(axis=1)

    def times_ht(Y: np.ndarray) -> np.ndarray:  # (B, N, G) -> (B, d*N, G)
        Yd = np.broadcast_to(Y[:, None], (B, d, N, Y.shape[2]))
        return _hankel_matmul(h_fft, Yd, N, nfft).reshape(B, d * N, -1)

    Q = np.linalg.qr(times_h(np.broadcast_to(omega, (B, d * N, G))))[0]
    for _ in range(n_iter):
        Q_tilde = np.linalg.qr(times_ht(Q))[0]
        Q = np.linalg.qr(times_h(Q_tilde))[0]
    small = np.swapaxes(times_ht(Q), 1, 2)  # B = Q^T [H_1, ..., H_d]
    U_small = np.linalg.svd(small, full_matrices=False)[0]
    return np.matmul(Q, U_small[:, :, :rank])


def sst_scores(
    X: np.ndarray,
    window_size: int,
    lag: int,
    rank: int,
    method: str = "fft_rsvd",
    oversampling: int = 10,
    n_iter: int = 3,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Change-point score of the singular spectrum transformation.

    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints, n_channels)
        Series, time along the first axis.
    window_size : int
        Window ``N``: length of the lagged vectors and number of them in a
        Hankel matrix (``n = w`` in Ide & Tsuda 2007).
    lag : int
        Lag ``gamma`` between the past and the future Hankel matrices.
    rank : int
        Dimension ``r`` of the past subspace.
    method : {"fft_rsvd", "exact"}, default="fft_rsvd"
        Randomised SVD with FFT Hankel products, or full SVD.
    oversampling : int, default=10
        Oversampling ``p`` of the randomised SVD.
    n_iter : int, default=3
        Subspace iterations ``q`` of the randomised SVD.
    random_state : int, Generator or None, default=None
        Seed of the Gaussian test matrix of the randomised SVD.

    Returns
    -------
    np.ndarray of shape (n_timepoints,)
        ``z(t) = 1 - ||U(t)^T mu(t)||^2`` in ``[0, 1]``, where ``U(t)`` spans
        the ``rank`` leading left singular vectors of the Hankel matrix of
        ``X[t - 2N + 1 : t]`` and ``mu(t)`` is the leading one of the matrix
        lagged by ``lag``; ``0`` where the two matrices do not fit in the
        series (the first ``2N - 1`` points and the last ``lag - 1``).
    """
    X = _preprocess(X)
    T, d = X.shape
    N = int(window_size)
    scores = np.zeros(T)
    first, last = 2 * N - 1, T - lag  # positions with both matrices in the series
    if last < first:
        return scores
    ends = np.arange(first, last + lag + 1)  # every matrix used, past or future
    if method == "exact":
        per_item = N * d * N
    else:
        G = rank + oversampling
        rng = np.random.default_rng(random_state)
        omega = rng.standard_normal((d * N, G))
        per_item = d * (next_fast_len(2 * N - 1, real=True) // 2 + 1) * G
    batch = max(1, _BATCH_ELEMENTS // per_item)
    U = np.empty((ends.size, N, rank))
    for start in range(0, ends.size, batch):
        h = _hankel_windows(X, N, ends[start : start + batch])
        if method == "exact":
            U[start : start + batch] = _left_vectors_exact(h, N, rank)
        else:
            U[start : start + batch] = _left_vectors_rsvd(h, N, rank, omega, n_iter)
    n_scores = last - first + 1
    mu = U[lag : lag + n_scores, :, 0]
    proj = np.sum(np.einsum("bnr,bn->br", U[:n_scores], mu) ** 2, axis=1)
    scores[first : last + 1] = np.clip(1.0 - proj, 0.0, 1.0)
    return scores
