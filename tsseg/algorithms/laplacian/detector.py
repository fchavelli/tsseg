"""Laplacian segmentation detector (McFee & Ellis, ISMIR 2014)."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import Closed, Interval, ParamDef, StrOptions
from .laplacian import (
    cluster_frames,
    k_louvain,
    k_silhouette,
    laplacian_embedding,
)

__all__ = ["LaplacianDetector"]


class LaplacianDetector(BaseSegmenter):
    """State detection by spectral clustering of a recurrence graph of windows.

    The Laplacian structural decomposition of McFee & Ellis [1]_, designed for
    the repeated sections of songs, applied to windows of a time series. The
    series is cut into frames; each frame is linked to the frames that look
    like it (mutual nearest neighbours, anywhere in the series) and to its
    two neighbours in time. The frames are clustered by k-means on the
    smallest eigenvectors of the normalised Laplacian of this graph, so that
    a state is a set of frames that recur together and stay together in time.

    Parameters
    ----------
    n_states : int or None, default=None
        Number of states. ``None`` (unsupervised): ``k = round(sqrt(k_L *
        k_S))``, the geometric mean of the number of Louvain communities of
        the recurrence graph (``k_L``, tends to over-segment) and of the
        ``k`` with the best silhouette in the feature space (``k_S``, tends to
        under-segment).
    window_size : int, default=128
        Frame length in points; on a series shorter than ``4 * window_size``,
        frames of ``max(8, n_timepoints // 4)`` points instead.
    step : int, default=16
        Points between consecutive frames. The cost is quadratic (memory) and
        cubic (eigensolver) in the number of frames, about
        ``n_timepoints / step``.
    feature : {"spec", "raw"}, default="spec"
        Frame features: ``"spec"`` is the log-magnitude spectrum of each
        channel (``log(1 + |FFT|)``, with the signed log DC term), invariant
        to the phase of periodic patterns, with the signed log of the
        window sum (DC term); ``"raw"`` is the window itself.
    rec_smooth : int, default=9
        Size, in frames, of the median filter along the diagonals of the
        recurrence matrix (1: no filter).
    evec_smooth : int, default=9
        Size, in frames, of the median filter of the eigenvectors along time
        (1: no filter).
    max_states : int, default=20
        Largest number of states considered when ``n_states=None``.
    random_state : int or None, default=None
        Seed of k-means, of the Louvain node order and of the frames sampled
        for the silhouette.
    axis : int, default=0
        Time axis of the input.

    Attributes
    ----------
    n_states_ : int
        Number of states used for the last prediction (given or estimated).
    k_louvain_, k_silhouette_ : int or None
        The two estimates combined when ``n_states=None`` (``None`` otherwise,
        or when the series has fewer than 6 frames or only identical ones).
    eigenvalues_ : np.ndarray or None
        Smallest eigenvalues of the normalised Laplacian.

    Notes
    -----
    Steps (librosa example "Laplacian segmentation" [2]_, msaf ``scluster``
    defaults [3]_): per-channel standardisation; frames of ``window_size``
    points every ``step`` points; mutual k-NN recurrence affinity between
    frames (cosine distance, ``2 ceil(sqrt(N - 2 e + 1))`` neighbours, links
    within ``e = ceil(window_size / (2 step))`` frames excluded),
    ``exp(-d / b)`` with ``b`` the median distance of a frame to its farthest
    link; diagonal median filter of ``rec_smooth`` frames; path affinity
    between consecutive frames, ``exp(-||f_i - f_{i+1}||^2 / median)``;
    ``A = mu R + (1 - mu) P`` with ``mu`` the least-squares balance of the
    degrees ([1]_, Eqs. 6-7); smallest eigenvectors of the normalised
    Laplacian of ``A`` ([1]_, Eq. 10),
    median filtered over ``evec_smooth`` frames, rows normalised by their
    cumulative norm; k-means (10 initialisations) on the first ``k``. Each
    point takes the label of the frame whose centre is nearest.

    Departures from the paper (the first three as in the librosa example and
    msaf, the others tsseg's):

    - recurrence: cosine distance and ``exp(-d / b)`` instead of the Gaussian
      kernel ``exp(-||x_i - x_j||^2 / (2 sigma^2))`` ([1]_, Sec. 2.3);
      ``2 ceil(sqrt(N - 2 e + 1))`` neighbours instead of
      ``1 + ceil(2 log2 N)``, and an exclusion of ``e`` frames instead of 3
      beats ([1]_, Sec. 4.2); median filter of the weighted affinity along
      the diagonals instead of a majority vote over 17 frames on the binary
      graph, weighted afterwards ([1]_, Eqs. 2 and 8). The cosine ignores
      scale: with ``feature="raw"``, two states that differ only by their
      amplitude look alike, and with ``"spec"`` frames that are nearly
      constant are dominated by their DC term, so that two levels of the
      same sign look alike;
    - path: weights on the same features as the recurrence, with the median
      squared distance as bandwidth, instead of a Gaussian kernel on MFCC
      ([1]_, Eq. 9); all path weights are 1 when more than half of the
      consecutive frames are identical. As in librosa, two frames at
      distance exactly 0 are never a recurrence link;
    - eigenvectors median filtered over ``evec_smooth`` frames (not in the
      paper);
    - frame features: a spectrum per window instead of beat-synchronous
      constant-Q and MFCC features of audio;
    - **symmetrised recurrence** (tsseg): the diagonal median filter breaks
      the symmetry of the recurrence matrix, and the librosa example and msaf
      decompose the resulting non-symmetric matrix with ``eigh``, which reads
      a single triangle. tsseg uses ``(Rf + Rf.T) / 2`` and checks that the
      spectrum lies in ``[0, 2]``, with a fallback to other LAPACK drivers;
    - fixed step in points (tsseg), so that the smoothing lengths mean the
      same duration on every series (no cap on the number of frames);
    - unknown number of states (tsseg): the paper keeps, among ``m = 2..10``,
      the partition of largest label entropy whose mean segment lasts at
      least 10 s ([1]_, Sec. 3.3), a threshold in seconds that has no
      counterpart for an arbitrary series; msaf and the librosa example fix
      ``k``. The rule above was selected on held-out TSB-SEG series (Louvain
      graph with ``2 ceil(sqrt(N))`` neighbours, no diagonal filter, no path
      edges; silhouette with cosine distance, on at most 2000 frames);
    - degenerate inputs: a single state when there are fewer than 6 frames
      or all frames are identical (constant series); the path graph alone
      when no pair of frames is a recurrence link.

    Defaults ``feature="spec"``, ``window_size=128``, ``step=16`` were
    declared before an evaluation on the 592 series of TSB-SEG; with them,
    guided (true ``n_states``) SMS 0.717 per series against 0.692 for CLaP.
    The method is weaker on univariate series (TSSB) than CLaP.

    The recurrence, the median filters and the Louvain step are ported from
    librosa (ISC) and NetworkX (BSD-3) with identical outputs, without these
    dependencies (notices in the sub-package ``LICENSE``).

    References
    ----------
    .. [1] B. McFee and D. P. W. Ellis. Analyzing song structure with
       spectral clustering. In Proceedings of the 15th International Society
       for Music Information Retrieval Conference (ISMIR), pages 405-410,
       2014.
    .. [2] librosa 0.11.0, gallery example "Laplacian segmentation"
       (``plot_segmentation.py``), ISC licence.
    .. [3] msaf (Music Structure Analysis Framework), algorithm ``scluster``,
       https://github.com/urinieto/msaf, MIT licence.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms import LaplacianDetector
    >>> t = np.arange(3000)
    >>> X = np.where((t // 500) % 2 == 0, np.sin(2 * np.pi * t / 50),
    ...              np.sin(2 * np.pi * t / 12))
    >>> labels = LaplacianDetector(n_states=2, random_state=0).fit_predict(X)
    >>> labels.shape
    (3000,)
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": False,
        "detector_type": "state_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "n_states": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            nullable=True,
            description="Number of states (None: estimated).",
        ),
        "window_size": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="Frame length in points.",
        ),
        "step": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Points between consecutive frames.",
        ),
        "feature": ParamDef(
            constraint=StrOptions({"spec", "raw"}),
            description="Frame features: log-magnitude spectrum or raw window.",
        ),
        "rec_smooth": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Diagonal median filter of the recurrence, in frames.",
        ),
        "evec_smooth": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Median filter of the eigenvectors, in frames.",
        ),
        "max_states": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="Largest number of states when n_states is None.",
        ),
        "random_state": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            nullable=True,
            description="Seed of k-means, Louvain and the silhouette sample.",
        ),
    }

    def __init__(
        self,
        *,
        n_states: int | None = None,
        window_size: int = 128,
        step: int = 16,
        feature: str = "spec",
        rec_smooth: int = 9,
        evec_smooth: int = 9,
        max_states: int = 20,
        random_state: int | None = None,
        axis: int = 0,
    ) -> None:
        self.n_states = n_states
        self.window_size = window_size
        self.step = step
        self.feature = feature
        self.rec_smooth = rec_smooth
        self.evec_smooth = evec_smooth
        self.max_states = max_states
        self.random_state = random_state
        super().__init__(axis=axis)
        if random_state is None:
            self.set_tags(non_deterministic=True)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        sd = X.std(axis=0)
        X = (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)
        n_eig = max(int(self.max_states), int(self.n_states or 0)) + 1
        emb = laplacian_embedding(
            X,
            window_size=int(self.window_size),
            step=int(self.step),
            feature=self.feature,
            n_eig=n_eig,
            rec_smooth=int(self.rec_smooth),
            evec_smooth=int(self.evec_smooth),
        )
        self.eigenvalues_ = emb.eigenvalues
        self.k_louvain_ = self.k_silhouette_ = None
        if emb.eigenvectors is None:  # < 6 frames, or all frames identical
            self.n_states_ = 1
            return np.zeros(X.shape[0], dtype=np.int64)
        if self.n_states is not None:
            k = int(self.n_states)
        else:
            self.k_louvain_ = k_louvain(emb, int(self.max_states), self.random_state)
            self.k_silhouette_ = k_silhouette(
                emb, int(self.max_states), self.random_state
            )
            k = int(round(np.sqrt(self.k_louvain_ * self.k_silhouette_)))
        k = int(min(k, emb.eigenvectors.shape[1]))
        self.n_states_ = max(k, 1)
        frame_labels = cluster_frames(emb.eigenvectors, k, self.random_state)
        return np.asarray(frame_labels[emb.frame_of_point], dtype=np.int64)
