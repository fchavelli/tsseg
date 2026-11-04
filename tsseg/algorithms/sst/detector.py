"""Singular spectrum transformation (SST) change-point detector."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..base import BaseSegmenter
from ..param_schema import Closed, DataDependent, Interval, ParamDef, StrOptions
from .sst import sst_scores

__all__ = ["SSTDetector"]


class SSTDetector(BaseSegmenter):
    r"""Change-point detection by singular spectrum transformation (SST).

    At every time :math:`t`, the past Hankel matrix
    :math:`H_1(t) = [s(t - N), \dots, s(t - 1)]`, whose columns are the
    lagged vectors :math:`s(j) = (x_{j - N + 1}, \dots, x_j)`, is compared
    with the future one :math:`H_2(t) = H_1(t + \gamma)`. Their dominant
    left singular subspaces describe the local linear dynamics of the series;
    the score

    .. math::

        z(t) = 1 - \sum_{i=1}^{r} \left(u_i(t)^\top \mu(t)\right)^2 \in [0, 1]

    is the squared distance between the leading future direction
    :math:`\mu(t)` and the span of the :math:`r` leading past ones
    :math:`u_1(t), \dots, u_r(t)` [2]_. It rises when the shape of the
    signal changes (frequency or form of an oscillation) and, after the
    preprocessing (see Notes), at level shifts; changes of the variance or
    of the autocorrelation of noise are found late or not at all. Change
    points are the ``n_cps`` highest peaks of the score at least
    ``min_distance`` points apart.

    By default the singular vectors come from the randomised SVD with
    FFT-based Hankel products of [3]_ ("FFT rSVD SST"), which costs
    :math:`O(q (k + p) N \log N)` per time point instead of :math:`O(N^3)`.

    Parameters
    ----------
    window_size : int, default=50
        Window :math:`N`: length of the lagged vectors and number of columns
        of each Hankel matrix (Idé & Tsuda's :math:`n = w`, Sec. 2 of [2]_).
        A score needs ``2 * window_size - 1 + lag`` points, which is also the
        minimum series length. The papers fix no value; 50 is the value of
        the experiments of [2]_.
    lag : int or None, default=None
        Lag :math:`\gamma` between the past and future matrices; ``None``
        means ``window_size // 2``, the default of [2]_ (Sec. 2).
    rank : int, default=5
        Dimension :math:`r < N` of the past subspace: :math:`k = 5` in [3]_
        (Sec. IV-E, windows of 100 to 5,000 points); [2]_ recommends 3 or 4
        (Sec. 2) and uses 3 with :math:`N = 50` (Sec. 5).
    oversampling : int, default=10
        Oversampling :math:`p` of the randomised SVD (:math:`k + p = 15` in
        [3]_, Sec. IV-E). Unused by ``method="exact"``.
    n_iter : int, default=3
        Subspace iterations :math:`q` of the randomised SVD (about
        :math:`\log N` in [3]_, Sec. IV-E). Unused by ``method="exact"``.
    method : {"fft_rsvd", "exact"}, default="fft_rsvd"
        ``"fft_rsvd"``: randomised SVD with FFT Hankel products, the variant
        that [3]_ recommends (Sec. V-A); ``"exact"``: full SVD of every
        Hankel matrix, the reference of [3]_ (cubic in ``window_size``).
    n_cps : int or None, default=None
        Number of change points to return: the ``n_cps`` highest peaks of the
        score, or fewer if it has fewer peaks. Required: the papers give no
        rule to choose the change points without it, and ``None`` raises a
        ``ValueError`` in ``predict``.
    min_distance : int or None, default=None
        Minimum number of points between two change points: of two peaks
        closer than that, the higher one is kept. ``None`` means
        ``window_size``.
    random_state : int or None, default=None
        Seed of the Gaussian test matrix of the randomised SVD. ``None``
        draws a new one at every ``predict`` (the score then varies by the
        approximation error, see Notes).
    axis : int, default=0
        Time axis.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_timepoints,)
        Score of the last series passed to ``predict``; ``0`` where the two
        Hankel matrices do not fit in the series (the first
        ``2 * window_size - 1`` points and the last ``lag - 1``).

    Notes
    -----
    Reimplemented from the papers, without third-party code.

    - Geometry, score and its time index follow Idé & Tsuda [2]_ (Sec. 2,
      Eq. 2.3): :math:`z(t)` is placed at the first point after the past
      matrix, which is the change point reported for a peak at :math:`t`.
      Weber & Lenz [3]_ (Alg. 1) compute the same score with
      :math:`\delta = \gamma` but place it at the last point of the future
      matrix, about :math:`\gamma` points later. Idé & Inoue [1]_ (Sec. 3.1)
      also normalise the score series to unit area; not done here (same
      peaks). No paper says where a change lies relative to the peak.
      Measured with the defaults: a level shift peaks within a few points
      of the change, a frequency change up to about :math:`N` points after
      it, a change of the variance of white noise several :math:`N` after
      it.
    - Every channel is first brought to unit variance and mean 3 ([2]_,
      Sec. 5.1 and footnote 2: positive values make the leading singular
      value non-degenerate; [1]_, Sec. 4.2: the SVD is not invariant to
      translation). Without the offset the score of white noise is erratic
      (tsseg's measurement).
    - Multivariate series (tsseg's choice, as the SST papers are univariate):
      the Hankel matrices of the channels are concatenated column-wise,
      :math:`H = [H^{(1)}, \dots, H^{(d)}] \in \mathbb{R}^{N \times dN}`,
      so that all channels share one subspace, as multivariate SSA stacks
      the Page matrices of the channels (Alanqary, Alomar & Shah [4]_,
      Sec. 2, Eq. 4; Hankel variant in its footnote 1; multilevel Hankel
      matrices are also suggested in [3]_, Sec. V-A).
    - Randomised SVD: one Gaussian test matrix with ``rank + oversampling``
      columns is drawn per ``predict`` and used for every time point; the
      leading future direction is taken from the same decomposition (rank
      ``rank``) as the past subspace of the matrix :math:`\gamma` points
      earlier, which is at least as accurate as the rank-1 decomposition of
      [3]_. QR orthonormalisation is used at every subspace iteration ([3]_
      uses LU in the intermediate ones, for speed only: same subspace in
      exact arithmetic). [3]_ reports a mean absolute score error of 0.036
      against the exact SST on its synthetic series (Table 4), without the
      preprocessing above; with it, the error is much smaller (the constant
      offset widens the gap between the leading singular values).
    - Selection: the papers give the score only. In guided mode, the
      ``n_cps`` highest peaks are kept (tsseg's choice); there is no
      unsupervised mode. At the ends of the series, where the matrices do
      not fit, the score is 0 ([2]_ uses periodic boundaries instead), and
      the first or last valid point can be a peak.

    The cost is :math:`O(T \cdot q (k + p) d N \log N)` for a series of
    :math:`T` points and :math:`d` channels with ``"fft_rsvd"``, and
    :math:`O(T d N^3)` with ``"exact"``: about 5 s for 20,000 points with
    the defaults.

    References
    ----------
    .. [1] T. Idé, K. Inoue. Knowledge discovery from heterogeneous dynamic
       systems using change-point correlations. SDM 2005, 571-575.
       https://doi.org/10.1137/1.9781611972757.63
    .. [2] T. Idé, K. Tsuda. Change-point detection using Krylov subspace
       learning. SDM 2007, 515-520.
       https://doi.org/10.1137/1.9781611972771.54
    .. [3] L. Weber, R. Lenz. Accelerating singular spectrum transformation
       for scalable change point detection. IEEE Access 13:213556-213577,
       2025. https://doi.org/10.1109/ACCESS.2025.3640386
    .. [4] A. Alanqary, A. Alomar, D. Shah. Change point detection via
       multivariate singular spectrum analysis. NeurIPS 2021.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms import SSTDetector
    >>> t = np.arange(1000)
    >>> X = np.where(t < 500, np.sin(2 * np.pi * t / 40), np.sin(2 * np.pi * t / 10))
    >>> SSTDetector(n_cps=1, random_state=0).fit_predict(X)  # doctest: +SKIP
    array([...])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "window_size": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="Window N: lagged-vector length and Hankel columns.",
        ),
        "lag": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Lag between the past and future Hankel matrices.",
            nullable=True,
        ),
        "rank": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Dimension of the past subspace.",
        ),
        "oversampling": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Oversampling of the randomised SVD.",
        ),
        "n_iter": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Subspace iterations of the randomised SVD.",
        ),
        "method": ParamDef(
            constraint=StrOptions({"fft_rsvd", "exact"}),
            description="Randomised SVD with FFT Hankel products, or full SVD.",
        ),
        "n_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points to return (required).",
            nullable=True,
        ),
        "min_distance": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum number of points between change points.",
            nullable=True,
        ),
        "random_state": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Seed of the randomised SVD.",
            nullable=True,
        ),
        "_cross_constraints": [
            DataDependent(
                "rank < window_size",
                "rank must be smaller than window_size.",
            ),
            DataDependent(
                "2 * window_size - 1 + (window_size // 2 if lag is None else lag)"
                " <= n_samples",
                "The series needs 2 * window_size - 1 + lag points.",
            ),
        ],
    }

    def __init__(
        self,
        *,
        window_size: int = 50,
        lag: int | None = None,
        rank: int = 5,
        oversampling: int = 10,
        n_iter: int = 3,
        method: str = "fft_rsvd",
        n_cps: int | None = None,
        min_distance: int | None = None,
        random_state: int | None = None,
        axis: int = 0,
    ) -> None:
        self.window_size = window_size
        self.lag = lag
        self.rank = rank
        self.oversampling = oversampling
        self.n_iter = n_iter
        self.method = method
        self.n_cps = n_cps
        self.min_distance = min_distance
        self.random_state = random_state
        super().__init__(axis=axis)
        if method == "fft_rsvd" and random_state is None:
            self.set_tags(non_deterministic=True)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self.n_cps is None:
            raise ValueError(
                "SSTDetector needs n_cps: the SST papers give no rule to choose "
                "change points from the score without their number."
            )
        if self.rank >= self.window_size:
            # r < w (Ide & Tsuda 2007, Sec. 2): with r = w the past subspace is
            # the whole space and the score is 0 everywhere.
            raise ValueError("rank must be smaller than window_size.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        lag = self.window_size // 2 if self.lag is None else int(self.lag)
        lag = max(lag, 1)
        scores = sst_scores(
            X,
            window_size=int(self.window_size),
            lag=lag,
            rank=int(self.rank),
            method=self.method,
            oversampling=int(self.oversampling),
            n_iter=int(self.n_iter),
            random_state=self.random_state,
        )
        self.scores_ = scores
        first, last = 2 * int(self.window_size) - 1, X.shape[0] - lag
        if self.n_cps == 0 or last < first:
            return np.empty(0, dtype=np.int64)
        distance = self.window_size if self.min_distance is None else self.min_distance
        return _highest_peaks(scores[first : last + 1], self.n_cps, distance) + first


def _highest_peaks(curve: np.ndarray, n_peaks: int, distance: int) -> np.ndarray:
    """Indices of the ``n_peaks`` highest peaks at least ``distance`` apart.

    The curve is padded with ``-inf`` on both sides, so that its first and
    last points can be peaks.
    """
    padded = np.concatenate(([-np.inf], curve, [-np.inf]))
    peaks, props = find_peaks(padded, distance=int(distance), height=-np.inf)
    order = np.argsort(-props["peak_heights"], kind="stable")[:n_peaks]
    return np.sort(peaks[order] - 1).astype(np.int64)
