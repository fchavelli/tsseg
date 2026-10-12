"""Random forest change point detection (changeforest) wrapper."""

from __future__ import annotations

import heapq
import math
import warnings

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import Closed, Interval, Options, ParamDef, StrOptions

__all__ = ["ChangeForestDetector"]


def _import_changeforest():
    """Import the ``changeforest`` package (``pip install tsseg[changeforest]``)."""
    try:
        import changeforest
    except ImportError as exc:
        raise ImportError(
            "ChangeForestDetector requires changeforest: "
            "pip install tsseg[changeforest]"
        ) from exc
    return changeforest


class ChangeForestDetector(BaseSegmenter):
    """Classifier-based multiple change point detection (changeforest).

    changeforest (Londschien, Bühlmann and Kovács, 2023) detects changes in
    the joint distribution of a multivariate series without a parametric
    model. To score a split ``t`` of a segment ``(u, v]``, it trains a
    classifier to tell observations before ``t`` from those after it and
    uses the out-of-bag class probabilities to compute a classifier
    log-likelihood ratio (the *gain*). A two-step search fits the classifier
    at the 1/4, 1/2 and 3/4 quantiles of the segment and maximises the
    resulting approximate gain curves, then refits the classifier at the
    best of these guesses and maximises the new gain curve over all
    candidate splits. A pseudo-permutation test of the first-step gains
    decides whether the best split is kept; kept splits are recursed on by
    binary segmentation.

    This detector wraps the official implementation, the ``changeforest``
    package (Rust, BSD-3-Clause), installed with
    ``pip install tsseg[changeforest]``.

    Parameters
    ----------
    method : {"random_forest", "knn", "change_in_mean"}, default="random_forest"
        Gain used to score splits: the random forest classifier
        log-likelihood ratio (the paper's method), the k-nearest-neighbour
        classifier variant ("changekNN" in the paper), or the parametric
        change-in-mean gain. Only the random forest is invariant to the
        scale of the channels; see Notes.
    segmentation : {"bs", "sbs", "wbs"}, default="bs"
        Search over segments: binary segmentation (the paper's method),
        seeded binary segmentation (Kovács et al., 2023) or wild binary
        segmentation (Fryzlewicz, 2014). The paper does not evaluate the
        last two with its model selection, which over-segments with them;
        see Notes.
    minimal_relative_segment_length : float, default=0.01
        Minimal segment length as a fraction ``delta`` of the series length:
        every segment has at least ``ceil(delta * n)`` observations. The
        paper's main simulations use 0.01.
    model_selection_alpha : float, default=0.02
        Threshold on the p-value of the pseudo-permutation test
        (classifier-based methods, unsupervised mode): a split is kept when
        ``(1 + #{permuted gains >= observed gain}) / (B + 1) <= alpha``. The
        paper calls it a tuning parameter rather than a valid significance
        level (Section 3.4); at 0.02, 2.5 to 5 % of its homogeneous series
        get at least one change point (Section 4.7, Table 3).
    model_selection_n_permutations : int, default=199
        Number ``B`` of permutations of the test.
    minimal_gain_to_split : float or None, default=None
        Minimal gain for a split to be kept (``method="change_in_mean"``,
        unsupervised mode). ``None`` uses the BIC-motivated value
        ``log(n) * (d + 1)``.
    number_of_wild_segments : int, default=100
        Number of random intervals drawn by ``segmentation="wbs"``.
    seeded_segments_alpha : float, default=1/sqrt(2)
        Decay of the seeded intervals of ``segmentation="sbs"``, in
        ``(0, 1)`` as in the package (Kovács et al. take it in ``[1/2, 1)``);
        values close to 1 give many intervals.
    random_forest_n_estimators : int, default=100
        Number of trees of each random forest.
    random_forest_max_depth : int or None, default=8
        Maximal depth of the trees; ``None`` grows them fully.
    random_forest_max_features : int, "sqrt" or None, default="sqrt"
        Number of features tried at each split of a tree: ``"sqrt"`` uses
        ``floor(sqrt(d))``, ``None`` all of them, an integer that many.
    random_forest_n_jobs : int, default=-1
        Number of threads used to grow the forests; -1 uses all cores.
    random_state : int, default=0
        Seed of the forests, of the permutation test and of the random
        intervals of ``segmentation="wbs"``. The output is deterministic for
        a given seed.
    n_cps : int or None, default=None
        Number of change points to return (semi-supervised mode). ``None``
        lets the permutation test (or ``minimal_gain_to_split``) choose. See
        Notes.
    axis : int, default=0
        Time axis of the input array.

    Notes
    -----
    Unsupervised mode (``n_cps=None``) returns the significant split points
    of the package's segmentation tree, exactly as ``changeforest`` does.

    The method assumes independent observations (Section 2 of the paper).
    On autocorrelated series it over-segments: three stationary AR(1)
    series with coefficient 0.9 (2000 points, 3 channels, no change) give 63
    to 71 change points each. Section 5 of the paper suggests adding lagged
    observations as extra channels when the change lies in the serial
    dependence.

    The random forest gain is invariant to the scale of each channel. The
    other two gains are not: ``"knn"`` uses Euclidean distances, and the
    ``"change_in_mean"`` gain (Section 3.1) and its threshold
    ``log(n) * (d + 1)`` assume noise of unit variance, so that i.i.d.
    Gaussian noise of standard deviation 3 (600 points, 5 channels) gives
    about 50 change points. Standardise the channels first, for instance
    by the median absolute deviation of their consecutive differences, as
    the paper does for its data sets (Section 4.2).

    The model selection is designed for binary segmentation. Section 5 of
    the paper notes that pairing it with seeded or wild binary segmentation
    would require changes to avoid overfitting, and the paper does not
    evaluate these combinations. On 20 series of i.i.d. Gaussian noise (600
    points, 5 channels), ``"bs"`` returned no change point, while ``"sbs"``
    and ``"wbs"`` returned at least one on 11 and 9 of them.

    Cost: the random forest needs a few forests per segment, each nearly
    linear in ``n`` for a fixed depth (Section 3.3). The ``"knn"`` gain
    builds an ``n x n`` distance matrix and its row-wise ordering, about
    ``16 n^2`` bytes (6.4 GB for ``n = 20000``).

    The package has no option for a fixed number of change points. With
    ``n_cps=K`` this detector runs binary segmentation best-first: it splits
    the whole series at its best split, then repeatedly splits the current
    segment whose best split has the largest gain, until ``K`` change points
    are found. The best split and the gain of a segment are those the
    package computes for that node of its tree (same classifier, two-step
    search, seed and minimal segment length ``ceil(delta * n)``); the
    significance test and ``minimal_gain_to_split`` are ignored. With
    ``segmentation="sbs"`` or ``"wbs"``, the seeded or random intervals are
    drawn inside each segment rather than once over the whole series. Fewer
    than ``K`` change points are returned, with a warning, when the segments
    become too short to split (the package splits a segment only when it
    has more than ``2 * ceil(delta * n)`` points). This guided mode is not
    part of the paper. Segments are ranked by their raw gain, which is
    negative on a homogeneous segment and decreases with its length, so
    change points beyond the true ones tend to fall in the shortest
    segments (a mean shift at 200 and 400 in 600 points with ``n_cps=4``
    gave ``[11, 18, 199, 400]``). With ``"sbs"`` or ``"wbs"``, every
    segment draws its own intervals, which is slower than ``"bs"`` (2 s
    against 0.04 s for 600 points and ``n_cps=2``).

    The hyperparameter defaults are those of the package (``Control`` in
    ``changeforest`` 1.2.1), which match the settings of the paper's main
    simulations (Section 4; Section 4.6 studies their influence).

    References
    ----------
    .. [1] M. Londschien, P. Bühlmann and S. Kovács. Random forests for change
       point detection. Journal of Machine Learning Research,
       24(216):1-45, 2023.
    .. [2] S. Kovács, P. Bühlmann, H. Li and A. Munk. Seeded binary
       segmentation: a general methodology for fast and optimal changepoint
       detection. Biometrika, 110(1):249-256, 2023.
    .. [3] P. Fryzlewicz. Wild binary segmentation for multiple change-point
       detection. The Annals of Statistics, 42(6):2243-2281, 2014.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms import ChangeForestDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([rng.normal(0, 1, (200, 3)), rng.normal(3, 1, (200, 3))])
    >>> ChangeForestDetector().fit_predict(X)  # doctest: +SKIP
    array([200])
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
        "python_dependencies": "changeforest",
    }

    _parameter_schema = {
        "method": ParamDef(
            constraint=StrOptions({"random_forest", "knn", "change_in_mean"}),
            description="Gain used to score splits.",
        ),
        "segmentation": ParamDef(
            constraint=StrOptions({"bs", "sbs", "wbs"}),
            description="Binary, seeded binary or wild binary segmentation.",
        ),
        "minimal_relative_segment_length": ParamDef(
            constraint=Interval(float, 0, 0.5, Closed.NEITHER),
            description="Minimal segment length, as a fraction of the series length.",
        ),
        "model_selection_alpha": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Threshold on the p-value of the pseudo-permutation test.",
        ),
        "model_selection_n_permutations": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of permutations of the permutation test.",
        ),
        "minimal_gain_to_split": ParamDef(
            constraint=Interval(float, None, None, Closed.NEITHER),
            description="Minimal gain to split (change_in_mean); None = BIC value.",
            nullable=True,
        ),
        "number_of_wild_segments": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of random intervals (wbs).",
        ),
        "seeded_segments_alpha": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Decay of the seeded intervals (sbs).",
        ),
        "random_forest_n_estimators": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of trees per forest.",
        ),
        "random_forest_max_depth": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Maximal tree depth (None = unlimited).",
            nullable=True,
        ),
        "random_forest_max_features": ParamDef(
            constraint=[
                StrOptions({"sqrt"}),
                Interval(int, 1, None, Closed.LEFT),
            ],
            description="Features tried per tree split ('sqrt', None = all, or int).",
            nullable=True,
        ),
        "random_forest_n_jobs": ParamDef(
            constraint=Options(int, set()),
            description="Number of threads (-1 = all cores).",
        ),
        "random_state": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Seed of the forests, permutations and random intervals.",
        ),
        "n_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points (None = permutation test).",
            nullable=True,
        ),
    }

    def __init__(
        self,
        *,
        method: str = "random_forest",
        segmentation: str = "bs",
        minimal_relative_segment_length: float = 0.01,
        model_selection_alpha: float = 0.02,
        model_selection_n_permutations: int = 199,
        minimal_gain_to_split: float | None = None,
        number_of_wild_segments: int = 100,
        seeded_segments_alpha: float = 1 / math.sqrt(2),
        random_forest_n_estimators: int = 100,
        random_forest_max_depth: int | None = 8,
        random_forest_max_features: int | str | None = "sqrt",
        random_forest_n_jobs: int = -1,
        random_state: int = 0,
        n_cps: int | None = None,
        axis: int = 0,
    ):
        self.method = method
        self.segmentation = segmentation
        self.minimal_relative_segment_length = minimal_relative_segment_length
        self.model_selection_alpha = model_selection_alpha
        self.model_selection_n_permutations = model_selection_n_permutations
        self.minimal_gain_to_split = minimal_gain_to_split
        self.number_of_wild_segments = number_of_wild_segments
        self.seeded_segments_alpha = seeded_segments_alpha
        self.random_forest_n_estimators = random_forest_n_estimators
        self.random_forest_max_depth = random_forest_max_depth
        self.random_forest_max_features = random_forest_max_features
        self.random_forest_n_jobs = random_forest_n_jobs
        self.random_state = random_state
        self.n_cps = n_cps
        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # BaseSegmenter overrides
    # ------------------------------------------------------------------

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        data = np.asarray(X, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        data = np.ascontiguousarray(data)
        n_samples = data.shape[0]

        if self.n_cps == 0:
            return np.empty(0, dtype=np.int64)

        # Minimal segment length, as computed by the package: ceil(delta * n).
        min_length = math.ceil(self.minimal_relative_segment_length * n_samples)

        if self.n_cps is None:
            # The package only splits segments longer than twice the minimal
            # length; on a shorter series, wild binary segmentation would never
            # stop drawing intervals.
            if n_samples <= 2 * min_length:
                return np.empty(0, dtype=np.int64)
            changeforest = _import_changeforest()
            result = changeforest.changeforest(
                data,
                self.method,
                self.segmentation,
                self._control(self.minimal_relative_segment_length),
            )
            cps = result.split_points()
        else:
            cps = self._predict_guided(data, int(self.n_cps), min_length)
        return np.asarray(sorted(cps), dtype=np.int64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _control(self, minimal_relative_segment_length, *, root_only=False):
        changeforest = _import_changeforest()
        kwargs = {
            "minimal_relative_segment_length": minimal_relative_segment_length,
            "model_selection_alpha": self.model_selection_alpha,
            "model_selection_n_permutations": self.model_selection_n_permutations,
            "number_of_wild_segments": self.number_of_wild_segments,
            "seeded_segments_alpha": self.seeded_segments_alpha,
            "seed": self.random_state,
            "random_forest_n_estimators": self.random_forest_n_estimators,
            "random_forest_max_depth": self.random_forest_max_depth,
            "random_forest_max_features": self.random_forest_max_features,
            "random_forest_n_jobs": self.random_forest_n_jobs,
        }
        if self.minimal_gain_to_split is not None:
            kwargs["minimal_gain_to_split"] = self.minimal_gain_to_split
        if root_only:
            # No split can pass model selection, so the package scores the
            # segment (best split and its gain) without recursing: the p-value
            # of a single permutation is at least 1/2, and no gain exceeds +inf.
            kwargs["model_selection_alpha"] = 0.25
            kwargs["model_selection_n_permutations"] = 1
            kwargs["minimal_gain_to_split"] = math.inf
        return changeforest.Control(**kwargs)

    def _best_split(self, data, start, stop, min_length):
        """Best split of ``data[start:stop]`` and its gain, as the package scores it.

        Returns ``None`` when the segment is too short to be split, that is
        when it has at most ``2 * min_length`` points (the package's rule).
        """
        length = stop - start
        if length <= 2 * min_length:
            return None
        changeforest = _import_changeforest()
        # The package uses ceil(delta * length) as minimal segment length; this
        # relative length gives the same ceil(delta * n) as on the whole series.
        relative = (min_length - 0.5) / length
        result = changeforest.changeforest(
            np.ascontiguousarray(data[start:stop]),
            self.method,
            self.segmentation,
            self._control(relative, root_only=True),
        )
        if result.best_split is None or result.max_gain is None:
            return None
        return start + int(result.best_split), float(result.max_gain)

    def _predict_guided(self, data, n_cps, min_length):
        """Best-first binary segmentation returning ``n_cps`` change points."""
        n_samples = data.shape[0]
        heap: list[tuple[float, int, int, int]] = []

        def push(start, stop):
            found = self._best_split(data, start, stop, min_length)
            if found is not None:
                split, gain = found
                heapq.heappush(heap, (-gain, start, stop, split))

        push(0, n_samples)
        cps: list[int] = []
        while heap and len(cps) < n_cps:
            _, start, stop, split = heapq.heappop(heap)
            cps.append(split)
            push(start, split)
            push(split, stop)

        if len(cps) < n_cps:
            warnings.warn(
                f"ChangeForestDetector found only {len(cps)} of the {n_cps} "
                "requested change points: the remaining segments are too short "
                f"to split (at most twice the minimal segment length, {min_length}).",
                UserWarning,
                stacklevel=3,
            )
        return cps
