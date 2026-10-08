"""RuLSIF change-point detector (Liu, Yamada, Collier and Sugiyama, 2013)."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from ..base import BaseSegmenter
from ..param_schema import Closed, DataDependent, HasType, Interval, ParamDef
from .rulsif import rulsif_scores

__all__ = ["RuLSIFDetector"]


class RuLSIFDetector(BaseSegmenter):
    r"""Change-point detection by relative density-ratio estimation (RuLSIF).

    Each split of the series is scored by how much the distribution of the
    ``n_subsequences`` subsequences before it differs from that of the
    ``n_subsequences`` subsequences after it. The difference is the
    symmetrised :math:`\alpha`-relative Pearson divergence
    :math:`\mathrm{PE}_\alpha(P \| P') + \mathrm{PE}_\alpha(P' \| P)`, each
    term estimated directly, without density estimation, by RuLSIF (a
    regularised least-squares fit of the relative density ratio with a
    Gaussian kernel model). Change points are peaks of this score.

    Parameters
    ----------
    subsequence_length : int, default=10
        Length ``k`` of the subsequences that form the samples; a
        multivariate subsequence concatenates all channels. Paper value.
    n_subsequences : int, default=50
        Number ``n`` of subsequences in each of the two compared sets. A
        score needs ``2 * n_subsequences + subsequence_length - 1`` points,
        which is also the minimum series length. Paper value.
    alpha : float, default=0.1
        Mixture weight :math:`\alpha \in [0, 1)` of the relative density
        ratio :math:`p / (\alpha p + (1 - \alpha) p')`; ``0`` gives the plain
        density ratio (uLSIF). Paper value.
    sigma_factors : tuple of float, default=(0.6, 0.8, 1.0, 1.2, 1.4)
        Candidate Gaussian kernel widths, as multiples of the median pairwise
        distance between the subsequences of the two sets. Paper values.
    lambdas : tuple of float, default=(1e-3, 1e-2, 1e-1, 1.0, 10.0)
        Candidate regularisation parameters. Paper values.
    n_folds : int, default=5
        Number of cross-validation folds used to choose the kernel width and
        the regularisation parameter, separately for every divergence
        estimate. Paper value.
    n_cps : int or None, default=None
        Number of change points to return (guided mode): the ``n_cps``
        highest peaks of the score at least ``min_distance`` apart, or fewer
        if the score has fewer peaks. ``None`` keeps the peaks above
        ``threshold``.
    threshold : float, default=3.0
        Minimum score of a peak in unsupervised mode. Not from the paper,
        which evaluates the score over all thresholds (ROC curves): about the
        99th percentile of the score on change-free series (white noise,
        AR(2), correlated Gaussian and noisy sine) with the default
        parameters. The divergence itself lies in ``[0, (1 - alpha) / alpha]``
        (``[0, 9]`` for the default ``alpha``); its estimate can fall slightly
        below 0.
    min_distance : int, default=20
        Minimum number of points between two change points. The paper merges
        alarms closer than 20 points when it evaluates the score (Section
        4.1).
    axis : int, default=0
        Time axis.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_timepoints,)
        Change-point score of the last series passed to ``predict``, ``nan``
        where it is undefined (about ``n_subsequences + subsequence_length /
        2`` points at each end).

    Notes
    -----
    Written for tsseg from the paper only; the authors' MATLAB code carries
    no licence and was not consulted. Choices the paper leaves open:

    - The score of the sets starting at ``t`` and ``t + n_subsequences`` is
      placed at ``t + n_subsequences + (subsequence_length - 1) // 2``,
      where the subsequences that straddle the boundary between the two sets
      are split evenly between them.
    - The median distance is taken over the ``2 * n_subsequences``
      subsequences of the two sets, for every split.
    - Cross-validation assigns sample ``i`` of each set to fold
      ``i % n_folds`` (deterministic) and keeps all samples of the
      numerator set as kernel centres, as in the paper's model (5). The
      held-out criterion is the RuLSIF squared loss :math:`J`.

    The cost is :math:`O(T \cdot F \cdot S \cdot L \cdot n^3)` for a series
    of length :math:`T`, :math:`F` folds, :math:`S` kernel widths and
    :math:`L` regularisation parameters: about 5 ms per point with the
    defaults, i.e. 25 s for 5,000 points.

    References
    ----------
    .. [1] S. Liu, M. Yamada, N. Collier, M. Sugiyama. Change-point detection
       in time-series data by relative density-ratio estimation. Neural
       Networks 43:72-83, 2013. https://doi.org/10.1016/j.neunet.2013.01.012,
       arXiv:1203.0453.
    .. [2] M. Yamada, T. Suzuki, T. Kanamori, H. Hachiya, M. Sugiyama.
       Relative density-ratio estimation for robust distribution comparison.
       Neural Computation 25(5):1324-1370, 2013.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms import RuLSIFDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 4, 300)])
    >>> RuLSIFDetector(n_cps=1).fit_predict(X)  # doctest: +SKIP
    array([300])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "subsequence_length": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Subsequence length k.",
        ),
        "n_subsequences": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="Number n of subsequences in each compared set.",
        ),
        "alpha": ParamDef(
            constraint=Interval(float, 0, 1, Closed.LEFT),
            description="Mixture weight of the relative density ratio.",
        ),
        "sigma_factors": ParamDef(
            constraint=HasType((tuple, list)),
            description="Kernel widths, as multiples of the median distance.",
        ),
        "lambdas": ParamDef(
            constraint=HasType((tuple, list)),
            description="Regularisation parameters.",
        ),
        "n_folds": ParamDef(
            constraint=Interval(int, 2, None, Closed.LEFT),
            description="Number of cross-validation folds.",
        ),
        "n_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points to return.",
            nullable=True,
        ),
        "threshold": ParamDef(
            constraint=Interval(float, None, None, Closed.NEITHER),
            description="Minimum peak score in unsupervised mode.",
        ),
        "min_distance": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum number of points between change points.",
        ),
        "_cross_constraints": [
            DataDependent(
                "2 * n_subsequences + subsequence_length - 1 <= n_samples",
                "The series needs 2 * n_subsequences + subsequence_length - 1 points.",
            ),
            DataDependent(
                "n_folds <= n_subsequences",
                "Each cross-validation fold needs at least one subsequence.",
            ),
        ],
    }

    def __init__(
        self,
        *,
        subsequence_length: int = 10,
        n_subsequences: int = 50,
        alpha: float = 0.1,
        sigma_factors: tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4),
        lambdas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0),
        n_folds: int = 5,
        n_cps: int | None = None,
        threshold: float = 3.0,
        min_distance: int = 20,
        axis: int = 0,
    ) -> None:
        self.subsequence_length = subsequence_length
        self.n_subsequences = n_subsequences
        self.alpha = alpha
        self.sigma_factors = sigma_factors
        self.lambdas = lambdas
        self.n_folds = n_folds
        self.n_cps = n_cps
        self.threshold = threshold
        self.min_distance = min_distance
        super().__init__(axis=axis)

    def _check_grids(self) -> None:
        for name in ("sigma_factors", "lambdas"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"{name} must be a non-empty sequence of numbers.")
        if np.any(np.asarray(self.sigma_factors, dtype=float) <= 0):
            raise ValueError("sigma_factors must be positive.")
        if np.any(np.asarray(self.lambdas, dtype=float) < 0):
            raise ValueError("lambdas must be non-negative.")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        self._check_grids()
        X = np.asarray(X, dtype=float)
        scores = rulsif_scores(
            X,
            k=self.subsequence_length,
            n=self.n_subsequences,
            alpha=self.alpha,
            sigma_factors=tuple(self.sigma_factors),
            lambdas=tuple(self.lambdas),
            n_folds=self.n_folds,
        )
        self.scores_ = scores
        return self._select_change_points(scores)

    def _select_change_points(self, scores: np.ndarray) -> np.ndarray:
        """Peaks of the score, by decreasing height, ``min_distance`` apart."""
        empty = np.empty(0, dtype=np.int64)
        if self.n_cps == 0:
            return empty
        valid = np.flatnonzero(~np.isnan(scores))
        if valid.size == 0:
            return empty
        start = int(valid[0])
        curve = scores[start : int(valid[-1]) + 1]
        peaks, props = find_peaks(curve, distance=self.min_distance, height=-np.inf)
        heights = props["peak_heights"]
        if self.n_cps is None:
            keep = heights > self.threshold
        else:
            keep = np.zeros(peaks.size, dtype=bool)
            keep[np.argsort(-heights, kind="stable")[: self.n_cps]] = True
        return np.sort(peaks[keep] + start).astype(np.int64)
