"""Laplacian structural decomposition of a time series (McFee & Ellis, 2014).

The recipe of the librosa gallery example "Laplacian segmentation" (ISC) and
of msaf's ``scluster`` (MIT), applied to windows of a time series:

1. frames of ``window_size`` points every ``step`` points, one feature vector
   per frame (log-magnitude spectrum per channel, or the raw window);
2. recurrence: mutual k-NN affinity between frames (cosine), then a median
   filter along the diagonals, then **symmetrised** (see Notes);
3. path: affinity between consecutive frames, ``exp(-||f_i - f_{i+1}||^2 / s)``
   with ``s`` the median of these squared distances;
4. ``A = mu R + (1 - mu) P``, with ``mu`` balancing the degrees of the two
   graphs (McFee & Ellis, Eq. 7);
5. smallest eigenvectors of the normalised Laplacian of ``A``, median filtered
   along time, rows normalised by the cumulative norm, k-means on the first
   ``k`` of them.

Notes
-----
librosa's diagonal median filter does not preserve symmetry (the zero padding
of the lag representation enters near the ends of every diagonal). The
librosa example and msaf pass the asymmetric matrix to ``scipy.linalg.eigh``,
which reads one triangle only, so the matrix decomposed is not the Laplacian
of a graph (negative eigenvalues, and a solver breakdown when a frame is
almost isolated). The filtered recurrence is symmetrised here, as
``(Rf + Rf.T) / 2``, and the spectrum is checked (in ``[0, 2]``, sorted),
with a fallback to other LAPACK drivers.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.sparse.csgraph
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from ._louvain import louvain_from_affinity
from ._recurrence import recurrence_affinity, timelag_median_filter

__all__ = [
    "LaplacianEmbedding",
    "cluster_frames",
    "frame_features",
    "k_louvain",
    "k_silhouette",
    "laplacian_embedding",
]

#: Fewer frames than this: no graph, a single state.
MIN_FRAMES = 6


class LaplacianEmbedding:
    """Result of :func:`laplacian_embedding`.

    Attributes
    ----------
    eigenvalues : np.ndarray or None
        Smallest eigenvalues of the normalised Laplacian, ascending.
    eigenvectors : np.ndarray or None
        Matching eigenvectors, median filtered along time, shape
        ``(n_frames, n_eig)``.
    features : np.ndarray
        Frame features, shape ``(n_frames, n_features)``.
    frame_of_point : np.ndarray
        Index of the frame whose centre is nearest to each point.
    window_size, step, exclusion : int
        Frame length and step actually used, and the recurrence exclusion
        band in frames.
    """

    def __init__(self, eigenvalues, eigenvectors, features, frame_of_point, **kw):
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.features = features
        self.frame_of_point = frame_of_point
        self.window_size = kw["window_size"]
        self.step = kw["step"]
        self.exclusion = kw["exclusion"]

    @property
    def n_frames(self) -> int:
        return int(self.features.shape[0])


def frame_features(W: np.ndarray, feature: str) -> np.ndarray:
    """Feature vector of each frame of ``W`` (shape ``(N, w, d)``).

    ``"spec"``: per channel, ``log(1 + |FFT|)`` of the non-constant frequencies
    and the signed ``log(1 + |DC|)`` (phase-invariant, the role the
    constant-Q spectrum plays for audio); ``"raw"``: the window itself.
    """
    N = W.shape[0]
    if feature == "raw":
        return W.reshape(N, -1)
    if feature == "spec":
        S = np.fft.rfft(W, axis=1)  # (N, w // 2 + 1, d)
        mag = np.log1p(np.abs(S[:, 1:, :]))
        dc = S[:, :1, :].real
        dc = np.sign(dc) * np.log1p(np.abs(dc))
        return np.concatenate([dc, mag], axis=1).reshape(N, -1)
    raise ValueError(f"Unknown feature {feature!r}: expected 'spec' or 'raw'.")


def _valid_spectrum(evals: np.ndarray) -> bool:
    """The spectrum of a normalised Laplacian lies in [0, 2], sorted."""
    return bool(
        np.all(np.isfinite(evals))
        and evals.min() > -1e-6
        and evals.max() < 2 + 1e-6
        and np.all(np.diff(evals) >= -1e-9)
    )


def _smallest_eigh(L: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray]:
    """``top`` smallest eigenpairs of the symmetric matrix ``L``.

    The subset driver (LAPACK ``evr``) can return values far outside [0, 2]
    without an error when many eigenvalues coincide (long constant
    stretches); fall back to the full ``evd``, then ``ev``, drivers.
    """
    try:
        evals, evecs = scipy.linalg.eigh(L, subset_by_index=[0, top - 1])
        if _valid_spectrum(evals):
            return evals, evecs
    except np.linalg.LinAlgError:
        pass
    for driver in ("evd", "ev"):
        try:
            evals, evecs = scipy.linalg.eigh(L, driver=driver)
        except np.linalg.LinAlgError:
            continue
        if _valid_spectrum(evals[:top]):
            return evals[:top], evecs[:, :top]
    raise np.linalg.LinAlgError("No eigensolver returned a valid Laplacian spectrum.")


def laplacian_embedding(
    X: np.ndarray,
    *,
    window_size: int = 128,
    step: int = 16,
    feature: str = "spec",
    n_eig: int = 21,
    rec_smooth: int = 9,
    evec_smooth: int = 9,
) -> LaplacianEmbedding:
    """Steps 1 to 5 up to the smoothed eigenvectors, shared by every ``k``.

    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints, n_channels)
        The series (standardised per channel by the detector).
    window_size : int, default=128
        Frame length in points; at most ``max(8, n_timepoints // 4)``.
    step : int, default=16
        Points between the starts of consecutive frames.
    feature : {"spec", "raw"}, default="spec"
        Frame features, see :func:`frame_features`.
    n_eig : int, default=21
        Number of eigenvectors (at most ``n_frames - 2``).
    rec_smooth, evec_smooth : int, default=9
        Sizes, in frames, of the diagonal median filter of the recurrence and
        of the median filter of the eigenvectors (msaf defaults).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    w = int(min(window_size, max(8, n // 4)))
    starts = np.arange(0, max(n - w + 1, 0), step)
    W = X[starts[:, None] + np.arange(w)[None, :]]  # (N, w, d)
    F = frame_features(W, feature)
    # a constant frame has a zero feature vector, for which the cosine is
    # undefined: give it a tiny constant vector instead
    F[~np.any(F, axis=1)] = 1e-8
    N = F.shape[0]
    # nearest frame centre (frame i is centred on i * step + w / 2)
    idx = np.clip(np.rint((np.arange(n) - w / 2.0) / step), 0, max(N - 1, 0))
    idx = idx.astype(int)
    exclusion = max(1, int(np.ceil(w / (2 * step))))
    exclusion = min(exclusion, max(1, (N - 1) // 4))
    meta = {"window_size": w, "step": int(step), "exclusion": exclusion}
    if N < MIN_FRAMES or not np.ptp(F, axis=0).any():
        # too few frames, or identical frames (constant series): no graph
        return LaplacianEmbedding(None, None, F, idx, **meta)

    R = recurrence_affinity(F, width=exclusion, metric="cosine")
    Rf = timelag_median_filter(R, rec_smooth)
    Rf = 0.5 * (Rf + Rf.T)  # see the module Notes

    path_d = np.sum(np.diff(F, axis=0) ** 2, axis=1)
    sigma = np.median(path_d)
    path_sim = np.exp(-path_d / sigma) if sigma > 0 else np.ones_like(path_d)
    P = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)
    if Rf.any():
        deg_p, deg_r = P.sum(1), Rf.sum(1)
        mu = deg_p.dot(deg_p + deg_r) / np.sum((deg_p + deg_r) ** 2)
        A = mu * Rf + (1 - mu) * P
    else:
        # no recurrence link at all (e.g. identical frames): mu would be 1 and
        # A empty; keep the path graph alone
        A = P

    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    top = min(n_eig, N - 2)
    evals, evecs = _smallest_eigh(L, top)
    V = scipy.ndimage.median_filter(evecs, size=(evec_smooth, 1))
    return LaplacianEmbedding(evals, V, F, idx, **meta)


def _normalised(V: np.ndarray, k: int) -> np.ndarray:
    """First ``k`` eigenvectors, rows divided by their cumulative norm."""
    cnorm = np.cumsum(V**2, axis=1) ** 0.5
    return V[:, :k] / np.maximum(cnorm[:, k - 1 : k], 1e-12)


def cluster_frames(
    V: np.ndarray, k: int, random_state: int | None = None
) -> np.ndarray:
    """Step 5: k-means (10 initialisations) on the first ``k`` eigenvectors."""
    k = int(min(k, V.shape[1]))
    if k <= 1:
        return np.zeros(V.shape[0], dtype=int)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    return km.fit_predict(_normalised(V, k))


def k_louvain(
    emb: LaplacianEmbedding, max_states: int = 20, random_state: int | None = None
) -> int:
    """Number of Louvain communities of the unfiltered recurrence graph.

    Graph: mutual k-NN affinity with ``2 ceil(sqrt(N))`` neighbours (at most
    ``N - 2 exclusion - 1``), without the diagonal median filter and without
    path edges; Louvain modularity at resolution 1. Capped at ``max_states``.
    """
    N, excl = emb.n_frames, emb.exclusion
    kn = min(2 * int(np.ceil(np.sqrt(N))), N - 2 * excl - 1)
    R = recurrence_affinity(emb.features, width=excl, k=kn, metric="cosine")
    n_comm = len(louvain_from_affinity(R, resolution=1.0, seed=random_state))
    return int(min(max(n_comm, 1), max_states))


def k_silhouette(
    emb: LaplacianEmbedding,
    max_states: int = 20,
    random_state: int | None = None,
    sample: int = 2000,
) -> int:
    """``k`` in ``2..max_states`` whose k-means labels (4 initialisations) have
    the best cosine silhouette in the frame-feature space (the same space for
    every ``k``), on at most ``sample`` frames."""
    V, F = emb.eigenvectors, emb.features
    N = V.shape[0]
    rng = np.random.default_rng(random_state)
    sub = np.sort(rng.choice(N, size=min(sample, N), replace=False))
    best_k, best_s = 2, -np.inf
    for k in range(2, min(max_states, V.shape[1]) + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=random_state)
        lab = km.fit_predict(_normalised(V, k))
        if len(np.unique(lab[sub])) < 2:
            continue
        s = silhouette_score(F[sub], lab[sub], metric="cosine")
        if s > best_s:
            best_k, best_s = k, s
    return best_k
