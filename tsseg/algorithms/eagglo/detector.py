"""E-Agglo: agglomerative clustering algorithm that preserves observation order."""

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)

__all__ = ["EAggloDetector"]


class EAggloDetector(BaseSegmenter):
    """
    Hierarchical agglomerative estimation of multiple change points.

    E-Agglo [1]_ starts from an initial partition of the series into contiguous
    clusters and repeatedly merges the pair of adjacent clusters that most
    increases a goodness-of-fit statistic, until a single cluster remains. The
    statistic is the sum, over pairs of adjacent clusters, of their scaled energy
    divergence :math:`\\hat Q(C_i, C_{i+1}; \\alpha)` (equation (13) in [1]_).
    The segmentation with the largest statistic along this merging sequence,
    optionally penalised, gives both the number and the locations of the change
    points. Unlike general purpose agglomerative clustering, only adjacent
    clusters are merged, so the time ordering is preserved.

    The energy divergence detects any change of distribution, without
    distributional assumptions beyond a finite ``alpha``-th moment, but assumes
    independent observations.

    This implementation is based on the aeon package, a port of the R package
    ``ecp`` [2]_.

    Parameters
    ----------
    member : array_like (default=None)
        Initial cluster of each time point: sorted labels, one per row of ``X``,
        each cluster a contiguous block. ``None`` puts every time point in its own
        cluster, as ``ecp`` does by default. Change points can only fall on the
        boundaries of the initial clusters. Blocks of a few samples are faster and
        less prone to over-segmentation (see Notes); [1]_ initialises its real data
        example with equally spaced blocks.
    alpha : float (default=1.0)
        Exponent of the distances in the energy divergence, in (0, 2], see
        equation (4) in [1]_. With ``alpha=2`` the divergence only compares the
        means of the clusters.
    penalty : str or callable or None (default=None)
        Added to the goodness-of-fit statistic of every step of the merging
        sequence before taking the maximum. A callable receives the segment
        boundaries of the step (the change points and the series length, plus 0
        unless the first and last clusters were merged) and returns a number.
        ``"len_penalty"`` subtracts the number of boundaries, ``"mean_diff_penalty"``
        adds the mean segment length. ``None`` applies no penalty.

    Attributes
    ----------
    merged_ : ndarray of shape (n_clusters - 1, 2)
        The pair of clusters merged at each step.
    gof_ : ndarray of shape (n_clusters,)
        Goodness-of-fit statistic of each step of the merging sequence, penalised
        when ``penalty`` is set; the returned segmentation is its argmax.
    cluster_ : ndarray of shape (n_timepoints,)
        Cluster of each time point in the returned segmentation.

    Notes
    -----
    As in ``ecp``, the divergences are V-statistics: the mean within-cluster
    distances include the zero distance of each point to itself, unlike the
    U-statistics of equation (5) in [1]_. Every boundary then adds about the mean
    within-cluster distance to the statistic, even without a change. From one
    cluster per time point and without a penalty, the maximum usually lies at a
    very fine segmentation, with many more change points than the true ones.
    Initial blocks help on independent observations, but on serially dependent
    series E-Agglo can over-segment whatever the initial partition; a penalty
    on the number of change points then has to be scaled to the statistic.

    Also as in ``ecp``, the first and last clusters count as adjacent: their
    divergence enters the statistic and they may be merged, giving a cluster that
    wraps around the end of the series (``cluster_`` then labels both ends alike).

    E-Agglo segments the series given to ``fit``: ``predict(X)`` returns those
    change points when ``X`` has the same values, and otherwise segments ``X``
    with a new fit, with a warning, leaving the fitted attributes unchanged. With
    ``member``, ``X`` must then have as many time points as ``member`` has labels.

    Time is :math:`O(n^2)` for a series of length :math:`n`. Memory is
    :math:`O(m^2)` for :math:`m` initial clusters, hence :math:`O(n^2)` with
    ``member=None``. Requires ``numba``.

    See Also
    --------
    BottomUpDetector : Greedy merges of adjacent segments under a segment cost,
        e.g. the RBF kernel cost, whose merge gain is also a two-sample
        (maximum mean discrepancy) statistic.

    References
    ----------
    .. [1] Matteson, David S., and Nicholas A. James. "A nonparametric approach for
       multiple change point analysis of multivariate data." Journal of the American
       Statistical Association 109.505 (2014): 334-345.
    .. [2] James, Nicholas A., and David S. Matteson. "ecp: An R package for
       nonparametric multiple change point analysis of multivariate data." arXiv preprint
       arXiv:1309.3295 (2013).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from tsseg.algorithms.eagglo.detector import EAggloDetector
    >>> rng = np.random.default_rng(0)
    >>> X = pd.DataFrame(rng.standard_normal((20, 2)))
    >>> model = EAggloDetector().fit(X)
    >>> model.cluster_.shape == (20,)
    True
    """

    _tags = {
        "X_inner_type": "pd.DataFrame",
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "python_dependencies": ["numba"],
        "capability:unsupervised": True,
        "capability:semi_supervised": False,
    }

    _parameter_schema = {
        "member": ParamDef(
            constraint=None,
            description="Initial cluster membership (None = one cluster per point).",
            nullable=True,
            ui_hidden=True,
        ),
        "alpha": ParamDef(
            constraint=Interval(float, 0, 2, Closed.RIGHT),
            description="Exponent for the divergence measure, in (0, 2].",
        ),
        "penalty": ParamDef(
            constraint=[
                StrOptions({"len_penalty", "mean_diff_penalty"}),
                HasType((Callable,)),
            ],
            description="Penalty function name or callable (None = no penalty).",
            nullable=True,
        ),
    }

    def __init__(
        self,
        member=None,
        alpha=1.0,
        penalty=None,
    ):
        self.member = member
        self.alpha = alpha
        self.penalty = penalty
        super().__init__(axis=0)

    def _fit(self, X, y=None):
        """Find optimally clustered segments.

        First, by determining which pairs of adjacent clusters will be merged_. Then,
        this process is repeated, recording the goodness-of-fit statistic at each step,
        until all observations belong to a single cluster. Finally, the estimated number
        of change points is estimated by the clustering that maximizes the goodness-of-
        fit statistic over the entire merging sequence.
        """
        from ._kernels import _update_distances

        # a copy: an array modified in place after fit is a new series
        self._X = X.copy()

        if self.alpha <= 0 or self.alpha > 2:
            raise ValueError(
                f"allowed values for 'alpha' are (0, 2], got: {self.alpha}"
            )

        self._initialize_params(X)

        # find which clusters optimize the gof_ and then update the distances
        for K in range(self.n_cluster - 1, 2 * self.n_cluster - 2):
            i, j = self._find_closest(K)

            _update_distances(
                i,
                j,
                K,
                self.merged_,
                self.n_cluster,
                self.distances,
                self.left,
                self.right,
                self.open,
                self.sizes,
                self.progression,
                self.lm,
            )

        def filter_na(i):
            return list(
                filter(
                    lambda v: v == v,
                    self.progression[i,],
                )
            )

        # penalize the gof_ statistic
        if self.penalty is not None:
            penalty_func = self._get_penalty_func()
            cps = [filter_na(i) for i in range(len(self.progression))]
            self.gof_ += list(map(penalty_func, cps))

        # get the set of change points for the "best" clustering
        idx = np.argmax(self.gof_)
        self._estimates = np.sort(filter_na(idx))

        # remove change point N+1 if a cyclic merger was performed
        self._estimates = (
            self._estimates[:-1] if self._estimates[0] != 0 else self._estimates
        )

        # create final membership vector
        def get_cluster(estimates):
            return np.repeat(
                range(len(np.diff(estimates))), np.diff(estimates).astype(int)
            )

        if self._estimates[0] == 0:
            self.cluster_ = get_cluster(self._estimates)
        else:
            tmp = get_cluster(np.append([0], self._estimates))
            self.cluster_ = np.append(tmp, np.zeros(X.shape[0] - len(tmp)))

        return self

    def _predict(self, X: pd.DataFrame, y=None):
        """Return detected change point indices.

        Parameters
        ----------
        X : pd.Dataframe
            Series to segment. If its values differ from those given to ``fit``,
            it is segmented with a new fit, with a warning, and the fitted
            attributes are left unchanged.
        y : default=None
            Ignored, for interface compatibility

        Returns
        -------
        np.ndarray
            Sorted 1-D integer array of change point indices (exclusive of 0
            and series length).

        Raises
        ------
        ValueError
            If ``member`` is set and ``X`` is a new series whose length differs
            from the number of labels in ``member``.
        """
        n = X.shape[0]
        if np.array_equal(X.to_numpy(), self._X.to_numpy()):
            estimates = self._estimates
        else:
            if self.member is not None and len(self.member) != n:
                raise ValueError(
                    f"member has {len(self.member)} labels, one per time point of "
                    f"the series given to fit, but X has {n} time points: use "
                    f"fit_predict(X) with a member of {n} labels."
                )
            warnings.warn(
                "Input data X differs from that given to fit(): segmenting X with "
                "a new fit, without updating the fitted attributes. To store them, "
                "use fit(X) or fit_predict(X).",
                UserWarning,
                stacklevel=3,
            )
            estimates = self.clone().fit(X)._estimates

        return np.array(
            sorted(int(cp) for cp in estimates if 0 < cp < n), dtype=np.int64
        )

    def _initialize_params(self, X: pd.DataFrame) -> None:
        """Initialize parameters and store to self."""
        from ._kernels import get_distance_matrix, get_distance_single

        self._member = np.array(
            self.member if self.member is not None else range(X.shape[0])
        )

        unique_labels = np.sort(np.unique(self._member))
        self.n_cluster = len(unique_labels)

        # relabel clusters to be consecutive numbers (when user specified)
        for i in range(self.n_cluster):
            self._member[np.where(self._member == unique_labels[i])[0]] = i

        # check if sorted
        if not all(sorted(self._member) == self._member):
            raise ValueError("'_member' should be sorted")

        self.sizes = np.zeros(2 * self.n_cluster)
        self.sizes[: self.n_cluster] = [
            sum(self._member == i) for i in range(self.n_cluster)
        ]  # calculate initial cluster sizes

        # array of between-within distances
        self.distances = np.empty((2 * self.n_cluster, 2 * self.n_cluster))

        # if there is an initial grouping...
        if self.member is not None:
            grouped = X.copy().set_index(self._member).groupby(level=0)
            within = grouped.apply(
                lambda x: get_distance_matrix(x.to_numpy(), x.to_numpy(), self.alpha)
            )

            for i, xi in grouped:
                xi_np = xi.to_numpy()
                self.distances[: self.n_cluster, i] = (
                    2
                    * grouped.apply(
                        lambda xj, _xi_np=xi_np: get_distance_matrix(
                            _xi_np,
                            xj.to_numpy(),
                            self.alpha,
                        )
                    )
                    - within[i]
                    - within
                )
        # else (no initial groupings)...
        else:
            X_num = X.to_numpy()
            for i in range(len(X)):
                for j in range(len(X)):
                    self.distances[i, j] = 2 * get_distance_single(
                        X_num[i], X_num[j], self.alpha
                    )

        np.fill_diagonal(self.distances, 0)

        # set up left and right neighbors
        # special case for clusters 0 and n_cluster-1 to allow for cyclic merging
        self.left = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.left[: self.n_cluster] = [
            i - 1 if i >= 1 else self.n_cluster - 1 for i in range(self.n_cluster)
        ]

        self.right = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.right[: self.n_cluster] = [
            i + 1 if i + 1 < self.n_cluster else 0 for i in range(self.n_cluster)
        ]

        # True means that a cluster has not been merged_
        self.open = np.ones(2 * self.n_cluster - 1, dtype=bool)

        # which clusters were merged_ at each step
        self.merged_ = np.empty((self.n_cluster - 1, 2))

        # set initial gof_ value
        self.gof_ = np.array(
            [
                sum(
                    self.distances[i, self.left[i]] + self.distances[i, self.right[i]]
                    for i in range(self.n_cluster)
                )
            ]
        )

        # change point progression
        self.progression = np.empty((self.n_cluster, self.n_cluster + 1))
        self.progression[0, :] = [
            sum(self.sizes[:i]) if i > 0 else 0 for i in range(self.n_cluster + 1)
        ]  # N + 1 for cyclic mergers

        # array to specify the starting point of a cluster
        self.lm = np.zeros(2 * self.n_cluster - 1, dtype=int)
        self.lm[: self.n_cluster] = range(self.n_cluster)

    def _find_closest(self, K: int) -> tuple[int, int]:
        """Determine which clusters will be merged_, for K clusters.

        Greedily optimize the goodness-of-fit statistic by merging the pair of adjacent
        clusters that results in the largest increase of the statistic's value.

        Parameters
        ----------
        K: int
            Number of clusters

        Returns
        -------
        result : Tuple[int, int]
            Tuple of left cluster and right cluster index values
        """
        from ._kernels import _gof_update

        best_fit = -1e10
        result = (0, 0)

        # iterate through each cluster to see how the gof_ value changes if merged_
        for i in range(K + 1):
            if self.open[i]:
                gof_ = _gof_update(
                    i, self.gof_, self.left, self.right, self.distances, self.sizes
                )

                if gof_ > best_fit:
                    best_fit = gof_
                    result = (i, self.right[i])

        self.gof_ = np.append(self.gof_, best_fit)
        return result

    def _get_penalty_func(self) -> Callable:  # sourcery skip: raise-specific-error
        """Define penalty function given (possibly string) input."""
        PENALTIES = {"len_penalty": len_penalty, "mean_diff_penalty": mean_diff_penalty}

        if callable(self.penalty):
            return self.penalty

        elif isinstance(self.penalty, str):
            if self.penalty in PENALTIES:
                return PENALTIES[self.penalty]

        raise Exception(
            f"'penalty' must be callable or {PENALTIES.keys()}, got {self.penalty}"
        )

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> list[dict]:
        """Test parameters."""
        return [
            {"alpha": 1.0, "penalty": None},
            {"alpha": 2.0, "penalty": "len_penalty"},
        ]


def len_penalty(x) -> int:
    """Penalize the goodness-of-fit statistic by the number of change points.

    ``x`` holds the segment boundaries of one step of the merging sequence.
    """
    return -len(x)


def mean_diff_penalty(x) -> float:
    """Reward the goodness-of-fit statistic by the mean segment length.

    Favors segmentations with larger segments. ``x`` holds the segment
    boundaries of one step of the merging sequence.
    """
    return float(np.mean(np.diff(np.sort(x))))


_KERNELS = frozenset(
    {
        "_gof_update",
        "_update_distances",
        "euclidean",
        "euclidean_matrix_to_matrix",
        "get_distance_matrix",
        "get_distance_single",
    }
)


def __getattr__(name: str):
    # The numba kernels moved to ``_kernels`` so that this module imports
    # without numba; they stay reachable from here.
    if name in _KERNELS:
        from . import _kernels

        return getattr(_kernels, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
