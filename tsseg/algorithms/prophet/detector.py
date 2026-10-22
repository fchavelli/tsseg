"""Prophet-based change-point detector exposed as a tsseg segmenter."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from prophet import Prophet

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    HasType,
    Interval,
    ParamDef,
    StrOptions,
)
from ..utils import multivariate_l2_norm

# pandas Timestamp (ns resolution) overflows around 2262-04-11.
# From start="2000-01-01" that leaves ~95 700 days with freq="D".
# For longer series we fall back to freq="s" to stay within bounds.
_MAX_DAILY_PERIODS = 90_000  # conservative safety margin


def _safe_date_range(n_periods: int) -> pd.DatetimeIndex:
    """Build a DatetimeIndex that fits within pandas Timestamp limits.

    Uses daily frequency when possible and falls back to seconds for very
    long series that would overflow the nanosecond Timestamp range.
    """
    freq = "D" if n_periods <= _MAX_DAILY_PERIODS else "s"
    return pd.date_range(start="2000-01-01", periods=n_periods, freq=freq)


def _candidate_indices(n_timepoints: int, n_candidates: int) -> np.ndarray:
    """Evenly spaced interior candidates, without the first and last samples."""
    if n_timepoints < 3 or n_candidates < 1:
        return np.empty(0, dtype=int)
    grid = np.linspace(0, n_timepoints - 1, n_candidates + 2).round().astype(int)[1:-1]
    return np.unique(grid[(grid > 0) & (grid < n_timepoints - 1)])


def _rate_changes(
    values: np.ndarray,
    ds: pd.DatetimeIndex,
    candidates: np.ndarray,
    changepoint_prior_scale: float,
    seasonality: bool,
) -> np.ndarray:
    """Rate change ``delta`` of Prophet's trend at each candidate (one channel)."""
    if np.ptp(values) == 0:
        return np.zeros(len(candidates))
    model = Prophet(
        changepoints=list(ds[candidates]),
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=seasonality,
        weekly_seasonality=seasonality,
        daily_seasonality=seasonality,
        uncertainty_samples=0,
    )
    model.fit(pd.DataFrame({"ds": ds, "y": values}))
    return np.asarray(model.params["delta"], dtype=float).reshape(-1)


def _block_means(values: np.ndarray, block: int) -> np.ndarray:
    """Means of consecutive blocks of ``block`` samples (the tail is dropped)."""
    if block <= 1:
        return values
    m = len(values) // block
    return values[: m * block].reshape(m, block).mean(axis=1)


def rate_change_profile(
    signal: np.ndarray,
    n_candidates: int,
    changepoint_prior_scale: float,
    seasonality: bool,
    multivariate_strategy: str,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate positions and Prophet's rate changes at each of them.

    ``signal`` has shape ``(n_timepoints, n_channels)``. Series longer than
    ``max_points`` are reduced to block means before the fit (Prophet's cost
    grows with the length times the number of candidates); the candidates are
    returned as indices of the full series. Returns ``(candidates, delta)``,
    ``delta`` of shape ``(n_models, n_candidates)``: one row per channel for
    ``"ensembling"``, one row for ``"l2"`` and univariate input.
    """
    n_timepoints, n_channels = signal.shape
    if n_channels > 1 and multivariate_strategy == "l2":
        channels = [multivariate_l2_norm(signal)]
    else:
        channels = [signal[:, c] for c in range(n_channels)]
    block = 1
    if max_points is not None and n_timepoints > max_points:
        block = int(np.ceil(n_timepoints / max_points))
    channels = [_block_means(np.asarray(v, dtype=float), block) for v in channels]
    n_reduced = len(channels[0])
    candidates = _candidate_indices(n_reduced, n_candidates)
    if candidates.size == 0:
        return candidates, np.empty((len(channels), 0))
    ds = _safe_date_range(n_reduced)
    # cmdstanpy logs every optimisation at INFO level.
    cmdstan_logger = logging.getLogger("cmdstanpy")
    disabled = cmdstan_logger.disabled
    cmdstan_logger.disabled = True
    try:
        delta = np.vstack(
            [
                _rate_changes(
                    values, ds, candidates, changepoint_prior_scale, seasonality
                )
                for values in channels
            ]
        )
    finally:
        cmdstan_logger.disabled = disabled
    return candidates * block, delta


def _relative_strength(delta: np.ndarray) -> np.ndarray:
    """|delta| over its median on each row (model), averaged over the rows.

    A noisy channel fitted with a loose prior has large rate changes
    everywhere; relative to its median they do not outweigh a channel with
    one clear change. The mean replaces a zero median (sparse fits), 1 a
    constant channel.
    """
    a = np.abs(np.asarray(delta, dtype=float))
    scale = np.median(a, axis=1, keepdims=True)
    mean = a.mean(axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, np.where(mean > 0, mean, 1.0))
    return (a / scale).mean(axis=0)


def _select(
    candidates: np.ndarray,
    strength: np.ndarray,
    n_cps: int | None,
    threshold: float,
    min_distance: int,
) -> np.ndarray:
    """Strongest candidates, at least ``min_distance`` apart (greedy NMS).

    With ``n_cps`` the ``n_cps`` strongest are kept, otherwise every candidate
    whose strength exceeds ``threshold``.
    """
    picked: list[int] = []
    for k in np.argsort(-strength, kind="stable"):
        if n_cps is not None and len(picked) >= n_cps:
            break
        if n_cps is None and strength[k] <= threshold:
            break
        if all(abs(candidates[k] - candidates[j]) >= min_distance for j in picked):
            picked.append(int(k))
    return np.sort(candidates[picked]).astype(int)


class ProphetDetector(BaseSegmenter):
    """Change points of Prophet's piecewise-linear trend.

    Prophet [1]_ models the trend as a piecewise-linear function whose rate
    may change at a set of potential change points, with a sparse (Laplace)
    prior on the rate changes ``delta``. The detector places ``n_candidates``
    potential change points evenly over the series and scores each one by
    its fitted ``|delta|`` divided by the median ``|delta|`` over the
    candidates (averaged over the channels with ``"ensembling"``). It returns
    the ``n_changepoints`` strongest when this number is given
    (semi-supervised), otherwise those whose score exceeds
    ``delta_threshold``, a relative form of TCPDBench's rule [2]_. A level shift
    falling between two candidates shows as two opposite rate changes on
    consecutive candidates, so two returned change points are at least
    ``tolerance`` apart (non-maximum suppression).

    The defaults were calibrated for change point detection on 24 series of
    TSB-SEG (3 per dataset): a prior scale far above Prophet's forecasting
    default (0.05) places the rate changes at the level shifts instead of
    spreading them over a smooth trend.

    Parameters
    ----------
    n_changepoints : int or None, default=None
        Number of change points to return. ``None`` selects them by
        ``delta_threshold``.
    n_changepoint_func : callable or None, default=None
        Callable returning ``n_changepoints`` from the series; overrides
        ``n_changepoints`` when given.
    n_candidates : int, default=200
        Number of potential change points, evenly spaced; their spacing
        bounds the localisation accuracy.
    changepoint_prior_scale : float, default=500.0
        Scale of Prophet's Laplace prior on the rate changes; larger values
        allow sharper trend changes (Prophet's own default is 0.05).
    delta_threshold : float, default=3.0
        Minimum score of a change point when ``n_changepoints`` is None: its
        ``|delta|`` relative to the median ``|delta|`` over the candidates.
    seasonality : bool, default=False
        Fit Prophet's yearly, weekly and daily seasonalities. The time index
        is synthetic, so they are off by default.
    multivariate_strategy : {"ensembling", "l2"}, default="l2"
        ``"ensembling"`` fits one model per channel and averages the scores
        of each candidate over the channels; ``"l2"`` fits one model on the
        L2 norm of the channels.
    tolerance : int or float, default=0.02
        Minimum distance between two returned change points, in samples, or
        as a fraction of the series length when below 1.
    max_points : int or None, default=2000
        Series longer than this are reduced to the means of consecutive
        blocks before the fit, which keeps Prophet's cost bounded; the
        candidates stay evenly spaced over the full series. ``None`` fits the
        full series.
    axis : int, default=0
        Time axis.

    References
    ----------
    .. [1] Taylor, S. J. and Letham, B. (2018). Forecasting at scale. The
       American Statistician, 72(1), 37-45.
    .. [2] van den Burg, G. J. J. and Williams, C. K. I. (2020). An evaluation
       of change point detection algorithms. arXiv:2003.06222.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
        "python_dependencies": ["prophet"],
    }

    _parameter_schema = {
        "n_changepoints": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of change points to return (None: thresholded).",
            nullable=True,
        ),
        "n_changepoint_func": ParamDef(
            constraint=HasType((Callable,)),
            description="Callable that determines n_changepoints from the series.",
            nullable=True,
            ui_hidden=True,
        ),
        "n_candidates": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of evenly spaced potential change points.",
        ),
        "changepoint_prior_scale": ParamDef(
            constraint=Interval(float, 0, None, Closed.NEITHER),
            description="Scale of the Laplace prior on the rate changes.",
        ),
        "delta_threshold": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Minimum |rate change| of a change point (unsupervised).",
        ),
        "seasonality": ParamDef(
            constraint=HasType((bool,)),
            description="Fit Prophet's yearly, weekly and daily seasonalities.",
        ),
        "multivariate_strategy": ParamDef(
            constraint=StrOptions({"ensembling", "l2"}),
            description="Strategy for multivariate series.",
        ),
        "tolerance": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Minimum distance between two change points.",
        ),
        "max_points": ParamDef(
            constraint=Interval(int, 10, None, Closed.LEFT),
            description="Longer series are reduced to block means before the fit.",
            nullable=True,
        ),
    }

    def __init__(
        self,
        *,
        n_changepoints: int | None = None,
        n_changepoint_func: Callable[[np.ndarray], int] | None = None,
        n_candidates: int = 200,
        changepoint_prior_scale: float = 500.0,
        delta_threshold: float = 3.0,
        seasonality: bool = False,
        multivariate_strategy: str = "l2",
        tolerance: int | float = 0.02,
        max_points: int | None = 2000,
        axis: int = 0,
    ) -> None:
        self.n_changepoints = (
            int(n_changepoints) if n_changepoints is not None else None
        )
        self.n_changepoint_func = n_changepoint_func
        self.n_candidates = n_candidates
        self.changepoint_prior_scale = changepoint_prior_scale
        self.delta_threshold = delta_threshold
        self.seasonality = seasonality
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        self.max_points = max_points
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        return self

    def _predict(self, X, axis=None):
        signal = np.asarray(X, dtype=float)
        if axis is not None and axis != self.axis:
            signal = np.moveaxis(signal, axis, self.axis)
        if signal.ndim == 1:
            signal = signal[:, None]
        n_timepoints, n_channels = signal.shape

        n_cps = (
            int(self.n_changepoint_func(signal))
            if self.n_changepoint_func is not None
            else self.n_changepoints
        )
        self.changepoint_strengths_ = np.empty(0, dtype=float)
        if n_cps == 0:
            self.changepoints_ = np.empty(0, dtype=int)
            return self.changepoints_
        candidates, delta = rate_change_profile(
            signal,
            self.n_candidates,
            self.changepoint_prior_scale,
            self.seasonality,
            self.multivariate_strategy,
            self.max_points,
        )
        if candidates.size == 0:
            self.changepoints_ = np.empty(0, dtype=int)
            return self.changepoints_
        strength = _relative_strength(delta)

        tol = self.tolerance
        min_distance = int(round(tol * n_timepoints)) if tol < 1 else int(tol)
        self.changepoints_ = _select(
            candidates,
            strength,
            n_cps,
            self.delta_threshold,
            max(min_distance, 1),
        )
        self.candidates_ = candidates
        self.changepoint_strengths_ = strength
        return self.changepoints_
