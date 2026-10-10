"""Seeded binary segmentation (Kovács et al., 2023), and wild binary
segmentation (Fryzlewicz, 2014) on the same machinery."""

from __future__ import annotations

import bisect
import warnings
from collections import deque

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    DataDependent,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)
from ..ruptures.detection.binseg import Binseg
from ._core import (
    greedy_path,
    narrowest_over_threshold,
    narrowest_path,
    seeded_intervals,
    ssic,
    wild_intervals,
)

__all__ = ["SeededBinSegDetector"]


def _segment_rss(csum, csum2, start, end):
    """Residual sum of squares of x[start:end] around its mean, per channel."""
    s1 = csum[end] - csum[start]
    return csum2[end] - csum2[start] - s1**2 / (end - start)


class SeededBinSegDetector(BaseSegmenter):
    """Seeded binary segmentation (SeedBS), or wild binary segmentation (WBS).

    Binary segmentation searches its best split on the whole series, then on
    each half, so that a change point lying between two others of opposite
    sign can stay hidden. Seeded binary segmentation instead computes the best
    split of every interval of a fixed, deterministic collection (the *seeded
    intervals*: layers of evenly spread, overlapping intervals whose length
    decays geometrically, ``O(n log n)`` points in all), then selects change
    points among these candidates (Kovács et al., 2023). Wild binary
    segmentation does the same on random intervals (Fryzlewicz, 2014).

    The gain of a split is the decrease of the cost when the interval is cut
    in two: for ``model="l2"`` the squared CUSUM statistic of the papers
    (summed over channels).

    Parameters
    ----------
    n_cps : int or None, default=None
        Number of change points (guided mode): the first ``n_cps`` of the
        greedy path, or, with ``selection="narrowest"``, the first solution of
        at least ``n_cps`` change points met while the threshold decreases,
        cut to the ``n_cps`` largest gains. Takes precedence over ``penalty``.
    penalty : float or None, default=None
        Threshold on the gain: a change point is kept when the gain of the
        interval that selects it exceeds ``penalty`` (for ``"l2"`` and noise
        of variance ``sigma^2``, ``(C sigma)^2 2 log n`` is the threshold
        ``C sigma sqrt(2 log n)`` of Fryzlewicz, 2014, Sec. 4, on the CUSUM).
        ``None``
        (with ``n_cps=None``): the strengthened Schwarz information criterion
        picks the model, as in the experiments of Kovács et al. (2023);
        ``model="l2"`` only.
    intervals : {"seeded", "wild"}, default="seeded"
        Seeded intervals (SeedBS) or ``n_intervals`` random ones (WBS).
    decay : float, default=2 ** -0.5
        Decay ``a`` in ``[1/2, 1)`` of the seeded interval lengths from one
        layer to the next; the value recommended by Kovács et al. (2023).
        Closer to 1: more intervals, slower and more accurate.
    n_intervals : int, default=5000
        Number of random intervals of ``intervals="wild"``, the value of
        Fryzlewicz (2014, Sec. 4.1) and of the experiments of Kovács et al.
        (2023); Baranowski et al. (2019, Sec. 3.6.1) recommend 10 000 for
        NOT. Both ends of each interval are drawn uniformly from the series, as
        in WBS (Fryzlewicz, 2014, Sec. 3.3); draws shorter than
        ``2 * min_size`` are dropped, not redrawn, so slightly fewer intervals
        are kept (NOT, Sec. 2.2, draws among the long enough ones only).
    selection : {"greedy", "narrowest"}, default="greedy"
        ``"greedy"``: the largest gain first, then all intervals containing it
        are dropped, and so on (as in WBS). ``"narrowest"``: among the
        intervals over the threshold, the narrowest first (narrowest over
        threshold, Baranowski et al., 2019), which the consistency theory of
        Kovács et al. (2023, Theorem 3) covers.
    alpha : float, default=1.01
        Exponent of the sSIC penalty ``k (d + 1)/2 log(n)^alpha``; 1 gives the
        Schwarz criterion. 1.01 is the value of Fryzlewicz (2014).
    max_cps : int, default=50
        Largest number of change points the sSIC considers (``K`` of
        Fryzlewicz, 2014, Sec. 4.2, which the criterion needs: along the whole
        path the residual variance tends to 0 and the largest model wins). The
        solution path is followed until it holds more; a series with more
        change points than ``max_cps`` gets ``max_cps`` at most. NOT
        evaluates every solution of its path with at most ``q_max = 25``
        change points (Baranowski et al., 2019, Sec. 3.6.2); here the path of
        ``selection="narrowest"``, whose solutions do not grow monotonically,
        stops at the first one with more than ``max_cps``, and smaller
        solutions further down the path are not evaluated.
    model : str, default="l2"
        Cost of the vendored ruptures (``"l2"``, ``"l1"``, ``"rbf"``,
        ``"linear"``, ``"normal"``, ``"cosine"``).
    min_size : int, default=1
        Minimum number of points on each side of a split; intervals shorter
        than ``2 * min_size`` have no admissible split and are dropped. ``1``
        is the ``m = 2`` of the experiments of Kovács et al. (2023). Above 1,
        the seeded intervals of the layers whose length is close to
        ``2 * min_size`` are kept when rounding makes them long enough;
        Kovács et al. (2023, Sec. 2.3) suggest instead stopping at an earlier
        layer when a minimal segment length is wanted.
    cost_params : dict or None, default=None
        Keyword arguments of the ruptures cost.
    backend : {"auto", "numba", "python"}, default="auto"
        Backend of the best split of the ``"l1"``, ``"l2"``, ``"rbf"`` and
        ``"cosine"`` costs, as for :class:`BinSegDetector`. Same result.
    random_state : int or None, default=None
        Seed of the random intervals (``intervals="wild"`` only).
    axis : int, default=0
        Time axis.

    Notes
    -----
    Indices follow tsseg: a change point is the first point of a new segment,
    the ``s`` of the papers' split ``(l, s] | (s, r]``. With
    ``min_size=1``, ``decay=2 ** -0.5``, ``selection="greedy"`` or
    ``"narrowest"`` and the sSIC, this is g-SeedBS or n-SeedBS of Kovács et
    al. (2023, Table 1). ``intervals="wild"`` with ``selection="greedy"`` is
    WBS with the sSIC, with the optional augmentation of Fryzlewicz (2014,
    Sec. 3.3) at the top level only: the whole series is always a candidate
    interval, but the segment between two selected change points is not
    added as a new candidate at each later step.

    Section, theorem and table numbers of Kovács et al. (2023) are those of
    arXiv:2002.06633v1, and those of Baranowski et al. (2019) those of its
    arXiv version.

    The sSIC (Fryzlewicz, 2014, Sec. 3.4, eq. (4)) assumes Gaussian noise of
    constant variance. For a multivariate series this implementation applies
    the general form of Baranowski et al. (2019, eq. (3.1): minus twice the
    log-likelihood plus ``log(n)^alpha`` per parameter, halved here) to
    Gaussian channels with their own variances, a change point counting its
    location and ``d`` means; it reduces to the univariate criterion for
    ``d = 1``. This multivariate model is not in the papers. The gain sums
    the channels as they are while the criterion weighs each by its own
    variance: when the channels have different noise levels, put them on the
    same noise level first, e.g. divide each by ``median(|diff(x)|) /
    (sqrt(2) * 0.6745)``, the noise estimate of Baranowski et al. (2019, Sec.
    2.1). A z-normalisation is not enough when a channel has large changes
    and little noise.

    Cost: the best split of every interval, ``O(n log n)`` points in all for
    the seeded intervals (``O(n_intervals * n)`` for the random ones), each in
    ``O(1)`` per point for ``"l2"``; the greedy selection then takes
    ``O(n log n)``. The narrowest selection with the sSIC recomputes its
    solution whenever a lower threshold changes it, up to ``max_cps`` change
    points: quadratic in the number of intervals in the worst case.

    References
    ----------
    .. [1] Kovács, S., Bühlmann, P., Li, H., Munk, A. (2023). Seeded binary
       segmentation: a general methodology for fast and optimal changepoint
       detection. Biometrika, 110(1), 249-256. arXiv:2002.06633.
    .. [2] Fryzlewicz, P. (2014). Wild binary segmentation for multiple
       change-point detection. The Annals of Statistics, 42(6), 2243-2281.
    .. [3] Baranowski, R., Chen, Y., Fryzlewicz, P. (2019). Narrowest-over-
       threshold detection of multiple change points and change-point-like
       features. Journal of the Royal Statistical Society B, 81(3), 649-672.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms import SeededBinSegDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([np.zeros(100), 3 * np.ones(100), np.zeros(100)])
    >>> X = X + rng.normal(size=300)
    >>> SeededBinSegDetector().fit_predict(X)
    array([100, 200])
    """

    _tags = {
        "fit_is_empty": True,
        "capability:univariate": True,
        "capability:multivariate": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "n_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points (guided mode).",
            nullable=True,
            group="stopping_criterion",
        ),
        "penalty": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Threshold on the gain; None: the sSIC picks the model.",
            nullable=True,
            group="stopping_criterion",
        ),
        "intervals": ParamDef(
            constraint=StrOptions({"seeded", "wild"}),
            description="Seeded (deterministic) or wild (random) intervals.",
        ),
        "decay": ParamDef(
            constraint=Interval(float, 0.5, 1, Closed.LEFT),
            description="Decay of the seeded interval lengths between layers.",
        ),
        "n_intervals": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of random intervals (intervals='wild').",
        ),
        "selection": ParamDef(
            constraint=StrOptions({"greedy", "narrowest"}),
            description="Largest gain first, or narrowest interval over the threshold.",
        ),
        "alpha": ParamDef(
            constraint=Interval(float, 1, None, Closed.LEFT),
            description="Exponent of the sSIC penalty.",
        ),
        "max_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Largest number of change points considered by the sSIC.",
        ),
        "model": ParamDef(
            constraint=StrOptions({"l1", "l2", "rbf", "linear", "normal", "cosine"}),
            description="Ruptures cost model name.",
        ),
        "min_size": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Minimum number of points on each side of a split.",
        ),
        "cost_params": ParamDef(
            constraint=HasType((dict,)),
            description="Extra kwargs for cost_factory.",
            nullable=True,
            ui_hidden=True,
        ),
        "backend": ParamDef(
            constraint=StrOptions({"auto", "numba", "python"}),
            description="Backend of the costs l1, l2, rbf and cosine.",
            ui_hidden=True,
        ),
        "random_state": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Seed of the random intervals.",
            nullable=True,
        ),
        "_cross_constraints": [
            DataDependent(
                "n_cps < n_samples",
                "n_cps must be less than the number of samples",
            ),
            DataDependent(
                "min_size * 2 <= n_samples",
                "min_size is too large for the series length",
            ),
        ],
    }

    def __init__(
        self,
        *,
        n_cps: int | None = None,
        penalty: float | None = None,
        intervals: str = "seeded",
        decay: float = 2**-0.5,
        n_intervals: int = 5000,
        selection: str = "greedy",
        alpha: float = 1.01,
        max_cps: int = 50,
        model: str = "l2",
        min_size: int = 1,
        cost_params: dict | None = None,
        backend: str = "auto",
        random_state: int | None = None,
        axis: int = 0,
    ) -> None:
        if n_cps is not None and penalty is not None:
            warnings.warn(
                "penalty is ignored when n_cps is provided", UserWarning, stacklevel=2
            )
        if n_cps is None and penalty is None and model != "l2":
            raise ValueError(
                "the sSIC (penalty=None) is defined for model='l2' only; "
                "give penalty or n_cps"
            )
        self.n_cps = n_cps
        self.penalty = penalty
        self.intervals = intervals
        self.decay = decay
        self.n_intervals = n_intervals
        self.selection = selection
        self.alpha = alpha
        self.max_cps = max_cps
        self.model = model
        self.min_size = min_size
        self.cost_params = cost_params
        self.backend = backend
        self.random_state = random_state
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        return self

    def _candidates(self, X: np.ndarray):
        """Bounds, best split and gain of every interval with a split."""
        solver = Binseg(
            model=self.model,
            min_size=self.min_size,
            jump=1,
            params=self.cost_params or None,
            backend=self.backend,
        )
        solver.fit(X)
        n = X.shape[0]
        min_length = 2 * solver.min_size
        if self.intervals == "seeded":
            bounds = seeded_intervals(n, self.decay, min_length)
        else:
            rng = np.random.default_rng(self.random_state)
            bounds = wild_intervals(n, self.n_intervals, min_length, rng)
        starts, ends, bkps, gains = [], [], [], []
        for start, end in bounds:
            bkp, gain, _ = solver.single_bkp(int(start), int(end))
            if bkp is None:
                continue
            starts.append(start)
            ends.append(end)
            bkps.append(bkp)
            gains.append(gain)
        self.n_intervals_ = len(bounds)
        self.total_length_ = int((bounds[:, 1] - bounds[:, 0]).sum())
        return (
            np.asarray(starts, dtype=np.int64),
            np.asarray(ends, dtype=np.int64),
            np.asarray(bkps, dtype=np.int64),
            np.asarray(gains, dtype=float),
        )

    def _predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        starts, ends, bkps, gains = self._candidates(X)
        greedy = self.selection == "greedy"

        if self.n_cps is not None:
            k = int(self.n_cps)
            if k == 0:
                found = []
            elif greedy:
                found = greedy_path(starts, ends, bkps, gains)[:k]
            else:
                # last solution of the path: the first with k change points
                # or more (or the one at the lowest threshold)
                path = narrowest_path(starts, ends, bkps, gains, stop_at=k)
                last = deque(path, maxlen=1)
                found = last[0] if last else []
                found = sorted(found, key=lambda item: -item[1])[:k]
        elif self.penalty is not None:
            if greedy:
                found = [
                    item
                    for item in greedy_path(starts, ends, bkps, gains)
                    if item[1] > self.penalty
                ]
            else:
                found = narrowest_over_threshold(
                    starts, ends, bkps, gains, float(self.penalty)
                )
        else:
            found = self._select_ssic(X, starts, ends, bkps, gains, greedy)
        return np.array(sorted(cp for cp, _ in found), dtype=np.int64)

    def _select_ssic(self, X, starts, ends, bkps, gains, greedy):
        """The model of smallest sSIC along the solution path."""
        centred = X - X.mean(axis=0)
        zeros = np.zeros((1, X.shape[1]))
        csum = np.vstack([zeros, np.cumsum(centred, axis=0)])
        csum2 = np.vstack([zeros, np.cumsum(centred**2, axis=0)])
        best, best_value = [], ssic(csum, csum2, [], self.alpha)
        limit = int(self.max_cps)
        if greedy:
            # nested models: each change point splits one segment, whose
            # residual sum of squares is updated in O(d)
            path = greedy_path(starts, ends, bkps, gains)[:limit]
            n, d = X.shape
            penalty = (d + 1) / 2 * np.log(n) ** self.alpha
            rss = np.maximum(csum2[-1] - csum[-1] ** 2 / n, 0.0)
            bounds = [0, n]
            for k, (cp, _) in enumerate(path, start=1):
                j = bisect.bisect_left(bounds, cp)
                left, right = bounds[j - 1], bounds[j]
                bounds.insert(j, cp)
                rss = rss - _segment_rss(csum, csum2, left, right)
                rss = rss + _segment_rss(csum, csum2, left, cp)
                rss = rss + _segment_rss(csum, csum2, cp, right)
                var = np.maximum(rss / n, np.finfo(float).tiny)
                value = float(n / 2 * np.log(var).sum() + k * penalty)
                if value < best_value:
                    best, best_value = path[:k], value
        else:
            path = narrowest_path(starts, ends, bkps, gains, stop_at=limit + 1)
            for solution in path:
                if len(solution) > limit:
                    break
                value = ssic(csum, csum2, [cp for cp, _ in solution], self.alpha)
                if value < best_value:
                    best, best_value = solution, value
        return best

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {"n_cps": 1}
