"""BEAST (Bayesian Estimator of Abrupt change, Seasonality, and Trend) detector."""

from __future__ import annotations

import warnings
from importlib import metadata

import numpy as np

from ..base import BaseSegmenter
from ..param_schema import (
    Closed,
    Interval,
    ParamDef,
    StrOptions,
)
from ..utils import aggregate_change_points, multivariate_l2_norm

__all__ = ["BeastDetector"]


_MIN_RBEAST = (0, 1, 25)
# Rbeast refuses an input array of fewer than three values ("The input data
# should be numeric and must be long enough") and then returns a result with
# an exception set, which Python raises as ``SystemError``.
_MIN_RBEAST_VALUES = 3


def _median_ncp(ncp_median, index: int, where: str) -> int | None:
    """Rbeast's median number of change points, or ``None`` if it gave none.

    Rbeast returns NaN for ``ncp_median`` when its run gives no estimate: on
    a series that is almost all NaN, every output is NaN. It also did in CI
    on a 6,000-point univariate series (not reproduced locally), where
    ``int(round(nan))`` raised ``ValueError``. The change points are then not
    capped by this count. A run in which every ``ncp_median`` is NaN is first
    rerun with Rbeast's SSE routines (see ``_all_nan_output``).
    """
    if ncp_median is None:
        return None
    values = np.asarray(ncp_median, dtype=float).ravel()
    if index >= len(values):
        return None
    value = values[index]
    if not np.isfinite(value):
        warnings.warn(
            f"Rbeast returned {value} for ncp_median ({where}): the number of "
            "change points is not capped, they are kept by cp_prob_threshold "
            "alone (none if every Rbeast output is NaN).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return int(round(float(value)))


def _all_nan_output(o) -> bool:
    """Whether every ``ncp_median`` of a Rbeast output is NaN: a failed run.

    ``ncp_median`` is the median number of change points of each component
    (``trend``, ``season``). Rbeast returns NaN for all of them, and for
    every other output, on an all-NaN series (which tsseg refuses) and, in
    CI, on a 6,000-point series in which it finds the change point on other
    machines (Rbeast issues #53, #56). An output without ``ncp_median`` is
    not counted as failed. With ``"native"``, each channel has its own run,
    and only the failure of every channel triggers the SSE rerun.
    """
    values = []
    for name in ("trend", "season"):
        ncp = getattr(getattr(o, name, None), "ncp_median", None)
        if ncp is not None:
            values.append(np.asarray(ncp, dtype=float).ravel())
    values = [v for v in values if v.size]
    return bool(values) and not any(np.isfinite(v).any() for v in values)


def _warn_sse_rerun() -> None:
    warnings.warn(
        "Rbeast returned NaN for every ncp_median, as when a run fails; "
        "rerunning it once with its SSE routines (cputype='sse'). On some CPUs, "
        "its default routines returned NaN for every output on series that "
        "other machines segment (Rbeast issues #53, #56).",
        RuntimeWarning,
        stacklevel=3,
    )


def _import_rbeast():
    """Import Rbeast (``pip install tsseg[beast]``), version 0.1.25 or later.

    Older releases never return on univariate series longer than about 4,600
    points (observed with 0.1.15).
    """
    try:
        import Rbeast as rb  # noqa: N813
    except ImportError as exc:
        raise ImportError(
            "BeastDetector requires Rbeast: pip install tsseg[beast]"
        ) from exc
    try:
        found = metadata.version("Rbeast")
    except metadata.PackageNotFoundError:
        found = None
    if found is not None:
        parts = tuple(int(p) for p in found.split(".")[:3] if p.isdigit())
        if parts < _MIN_RBEAST:
            raise ImportError(
                f"BeastDetector requires Rbeast >= 0.1.25 (found {found}): older "
                "releases hang on univariate series longer than about 4,600 points. "
                "pip install -U Rbeast"
            )
    return rb


class BeastDetector(BaseSegmenter):
    """Bayesian ensemble change-point detector using BEAST (Rbeast).

    BEAST decomposes a time series into trend + seasonality + noise via
    Bayesian model averaging over a large space of piecewise-linear trend
    models and piecewise-harmonic seasonal models. Change points are inferred
    from the posterior distribution, with each time step receiving a
    probability of being a change point.

    This detector wraps the ``Rbeast`` package (``pip install tsseg[beast]``,
    Rbeast 0.1.25 or later).

    Parameters
    ----------
    season : str, default="none"
        Seasonal model type: ``"none"`` for trend-only data; ``"harmonic"``,
        ``"dummy"`` or ``"svd"`` for data with periodic components (see
        ``period``). With a seasonal model, Rbeast 0.1.25 can kill the Python
        process (abort on a double free, or segmentation fault) instead of
        raising an error, depending on the length of the series, the period
        and the values: for instance ``"dummy"`` with ``period=2`` on 3 to 5
        points, or with ``period=12`` on 200 points of white noise. A direct
        call to Rbeast crashes the same way. The crash can also come after the
        call has returned. No crash was seen with ``"none"``. To keep the caller alive, run each
        call in a new child process, for instance with
        ``concurrent.futures.ProcessPoolExecutor``, which raises
        ``BrokenProcessPool`` when the child dies.
    tcp_minmax : tuple of int, default=(0, 10)
        Minimum and maximum number of trend change points. Rbeast returns at
        most the maximum per channel; a larger ``max_cps`` raises it (see
        ``max_cps``).
    torder_minmax : tuple of int, default=(0, 1)
        Minimum and maximum polynomial orders for the trend.
    tseg_minlength : int or None, default=None
        Minimum segment length for the trend component. ``None`` lets BEAST
        pick a default.
    scp_minmax : tuple of int, default=(0, 10)
        Minimum and maximum number of seasonal change points.
    sorder_minmax : tuple of int, default=(0, 5)
        Minimum and maximum harmonic orders for the seasonal component.
    sseg_minlength : int or None, default=None
        Minimum segment length for the seasonal component.
    period : float, default=NaN
        Period of the seasonal signal (e.g., 12 for monthly data with annual
        seasonality). Only used when ``season != "none"``.
    deltat : float, default=1.0
        Time interval between consecutive observations.
    mcmc_samples : int, default=8000
        Number of MCMC samples to collect.
    mcmc_burnin : int, default=200
        Number of initial samples to discard per chain.
    mcmc_chains : int, default=3
        Number of MCMC chains to run.
    mcmc_thin : int, default=5
        Thinning factor for the MCMC chains.
    mcmc_seed : int, default=1
        Seed of the MCMC sampler. ``0`` lets Rbeast pick a random seed, so the
        output changes from one call to the next. A fixed seed gives the same
        output on one machine, not necessarily on another: Rbeast picks its
        numerical routines by CPU instruction set (SSE, AVX2, AVX-512), and
        the sampler amplifies their rounding differences (Rbeast issue #53).
    cp_prob_threshold : float, default=0.1
        Minimum posterior probability of being a change point for a time
        step to be selected.
    max_cps : int or None, default=None
        Maximum number of change points to return (by decreasing probability).
        ``None`` returns all candidates above ``cp_prob_threshold`` (with
        ``"native"``, as many as a channel has, see Notes); ``0`` returns none.
        A ``max_cps`` above the maximum of ``tcp_minmax`` (and, with a
        seasonal component returned, of ``scp_minmax``) raises that maximum to
        ``max_cps``, since Rbeast never returns more change points per
        component and channel. On a univariate series, with ``"l2"``, and on each
        channel with ``"ensembling"``, the candidates kept are those with the
        largest occurrence probability (Rbeast's ``cpOccPr``) of ``component``
        at their time step; with ``"both"``, the larger of the trend and
        seasonal probabilities.
    component : str, default="trend"
        Which component's change points to return: ``"trend"`` or
        ``"season"`` or ``"both"``.
    multivariate_strategy : str, default="native"
        Strategy for multivariate inputs: ``"native"`` runs ``beast123`` on
        each channel, as Rbeast fits the columns of a 2D array (independently),
        and pools their candidates (see Notes); ``"l2"`` reduces channels via
        L2 norm; ``"ensembling"`` runs BEAST per channel and aggregates.
    tolerance : int or float, default=0
        Distance, in time steps, within which change points of a multivariate
        series count as one (a float below 1 is a fraction of the series
        length). With ``"native"``, candidates are taken by decreasing
        support (see Notes), and a candidate within ``tolerance`` of one
        already taken is dropped, on top of the merge of close candidates of
        different channels described in the Notes; ``0`` adds none. With
        ``"ensembling"``,
        the change points of all channels are grouped when within
        ``tolerance`` of the previous one, and each group gives its median
        (``max_cps`` keeps the largest groups); ``0`` groups identical time
        steps only. Not used with ``"l2"`` nor on a univariate series.
    axis : int, default=0
        Time axis of the input array.

    Notes
    -----
    Change points are read from Rbeast's ranked list of the most likely
    locations (``cp``, with probabilities ``cpPr``): those with a probability
    of at least ``cp_prob_threshold``, at most ``ncp_median`` of them
    (Rbeast's median number of change points) per component and, with
    ``"native"``, per channel. When Rbeast returns NaN for ``ncp_median``, a
    ``RuntimeWarning`` is raised and their number is not capped.

    With ``"native"``, the candidates of every channel (and, with ``"both"``,
    of both components) are pooled, and a time step found more than once
    keeps its largest ``cpPr``. Rbeast keeps the trend change points of one
    channel a minimum distance apart: ``tseg_minlength`` or, if None, 3 time
    steps (half the period with a seasonal component), at least the largest
    trend order plus one and at most ``n / 2 - 1``. Two candidates of
    different channels closer than this distance are taken as the same
    change, found by both channels: only one of them is returned. The same
    change found a few more time steps apart is returned twice, unless
    ``tolerance`` merges them.

    The pooled candidates are taken by decreasing support, the sum over the
    channels of the ``cpPr`` of the candidate of each channel at the same
    step, closer than this distance or within ``tolerance`` (a change seen
    by several channels comes first), then by decreasing ``cpPr``. Without
    ``max_cps``, as many are returned as a channel that changes has: the
    median number of candidates of the channels that have any, rounded half
    up. Their union would grow with the number of channels, each giving up to
    the maximum of ``tcp_minmax``. ``max_cps`` keeps that many, after the
    merges. A change seen by one channel only is returned, but the count
    assumes changes shared by the channels: when they change at different
    times, some of their changes are left out.

    With ``"native"``, each channel goes to ``beast123`` on its own. Given a
    2D array, Rbeast 0.1.25 fits its columns independently too, but in threads
    pinned to CPUs 0, 1, ... (``glue_code.c``), whatever CPUs the process may
    use. Where it may not use them (a job of a cluster scheduler, a
    container limited to some CPUs), the threads are not created: Rbeast
    returned unset outputs (no change point, or wrong ones) and corrupted the
    memory, so that the process was killed later. A single series runs in
    the calling thread, on any CPU.

    On some machines, Rbeast 0.1.25 returned NaN for every output on a
    6,000-point series in which it finds the change point elsewhere; Rbeast
    issue #56 reports wrong outputs on Intel Xeon CPUs. When ``ncp_median`` is
    NaN in every component and every channel, the detector raises a
    ``RuntimeWarning`` and reruns Rbeast once with its SSE routines
    (``cputype="sse"``, which took 1.5 to 2 times as long as the default
    routines on an AVX2 CPU). If the rerun returns NaN too, the detector
    returns no change point. Other runs are left as they are.

    Rbeast refuses an input of fewer than three values. A series of two
    points, univariate or multivariate, is therefore not passed to Rbeast and
    gets no change point, like any series of fewer than two points.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": True,
        "returns_dense": True,
        "detector_type": "change_point_detection",
        "python_dependencies": ["Rbeast"],
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    _parameter_schema = {
        "season": ParamDef(
            constraint=StrOptions({"none", "harmonic", "dummy", "svd"}),
            description="Seasonal model type.",
        ),
        "tcp_minmax": ParamDef(
            constraint=None,
            description="(min, max) number of trend change points.",
        ),
        "torder_minmax": ParamDef(
            constraint=None,
            description="(min, max) polynomial orders for the trend.",
        ),
        "tseg_minlength": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Min segment length for the trend component.",
            nullable=True,
        ),
        "scp_minmax": ParamDef(
            constraint=None,
            description="(min, max) number of seasonal change points.",
        ),
        "sorder_minmax": ParamDef(
            constraint=None,
            description="(min, max) harmonic orders for the seasonal component.",
        ),
        "sseg_minlength": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Min segment length for the seasonal component.",
            nullable=True,
        ),
        "mcmc_samples": ParamDef(
            constraint=Interval(int, 100, None, Closed.LEFT),
            description="Number of MCMC samples to collect.",
        ),
        "mcmc_burnin": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Number of burn-in samples per chain.",
        ),
        "mcmc_chains": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Number of MCMC chains.",
        ),
        "mcmc_thin": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
            description="Thinning factor for MCMC chains.",
        ),
        "mcmc_seed": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="MCMC seed (0 = random seed, not reproducible).",
        ),
        "cp_prob_threshold": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Min posterior probability to accept a change point.",
        ),
        "max_cps": ParamDef(
            constraint=Interval(int, 0, None, Closed.LEFT),
            description="Maximum number of change points to return.",
            nullable=True,
        ),
        "component": ParamDef(
            constraint=StrOptions({"trend", "season", "both"}),
            description="Which component's change points to return.",
        ),
        "multivariate_strategy": ParamDef(
            constraint=StrOptions({"native", "l2", "ensembling"}),
            description="Strategy for multivariate data.",
        ),
        "tolerance": ParamDef(
            constraint=Interval(float, 0, None, Closed.LEFT),
            description="Tolerance for merging close CPs (native, ensembling).",
        ),
    }

    def __init__(
        self,
        season: str = "none",
        tcp_minmax: tuple[int, int] = (0, 10),
        torder_minmax: tuple[int, int] = (0, 1),
        tseg_minlength: int | None = None,
        scp_minmax: tuple[int, int] = (0, 10),
        sorder_minmax: tuple[int, int] = (0, 5),
        sseg_minlength: int | None = None,
        period: float = float("nan"),
        deltat: float = 1.0,
        mcmc_samples: int = 8000,
        mcmc_burnin: int = 200,
        mcmc_chains: int = 3,
        mcmc_thin: int = 5,
        mcmc_seed: int = 1,
        cp_prob_threshold: float = 0.1,
        max_cps: int | None = None,
        component: str = "trend",
        multivariate_strategy: str = "native",
        tolerance: int | float = 0,
        axis: int = 0,
    ) -> None:
        self.season = season
        self.tcp_minmax = tuple(tcp_minmax)
        self.torder_minmax = tuple(torder_minmax)
        self.tseg_minlength = tseg_minlength
        self.scp_minmax = tuple(scp_minmax)
        self.sorder_minmax = tuple(sorder_minmax)
        self.sseg_minlength = sseg_minlength
        self.period = float(period)
        self.deltat = float(deltat)
        self.mcmc_samples = int(mcmc_samples)
        self.mcmc_burnin = int(mcmc_burnin)
        self.mcmc_chains = int(mcmc_chains)
        self.mcmc_thin = int(mcmc_thin)
        self.mcmc_seed = int(mcmc_seed)
        self.cp_prob_threshold = float(cp_prob_threshold)
        self.max_cps = max_cps if max_cps is None else int(max_cps)
        self.component = component
        self.multivariate_strategy = multivariate_strategy
        self.tolerance = tolerance
        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # BaseSegmenter overrides
    # ------------------------------------------------------------------

    def _fit(self, X, y=None):
        return self

    def _predict(self, X):
        data = np.asarray(X, dtype=float)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        n_samples, n_channels = data.shape

        if n_samples < 2:
            return np.empty(0, dtype=np.int64)

        if n_channels == 1:
            return self._run_beast_univariate(data.ravel())

        if self.multivariate_strategy == "native":
            return self._run_beast_multivariate(data)
        elif self.multivariate_strategy == "ensembling":
            return self._predict_ensembling(data)
        else:  # l2
            signal = multivariate_l2_norm(data)
            return self._run_beast_univariate(signal)

    def _predict_ensembling(self, data: np.ndarray) -> np.ndarray:
        """Run BEAST independently on each channel and aggregate results."""
        n_samples, n_channels = data.shape
        all_cps: list[int] = []

        for d in range(n_channels):
            cps = self._run_beast_univariate(data[:, d])
            all_cps.extend(cps.tolist())

        if not all_cps:
            return np.empty(0, dtype=np.int64)

        n_cp = self.max_cps if self.max_cps is not None else len(all_cps)
        return aggregate_change_points(
            all_cps,
            n_cp=n_cp,
            tolerance=self.tolerance,
            signal_len=n_samples,
        )

    def _get_mcmc_kwargs(self) -> dict:
        """Return common MCMC keyword arguments for beast() / beast123()."""
        return dict(
            start=0.0,  # time of the first sample: CP times / deltat are indices
            season=self.season,
            deltat=self.deltat,
            period=self.period,
            tcp_minmax=list(self._tcp_minmax()),
            torder_minmax=list(self.torder_minmax),
            tseg_minlength=self.tseg_minlength,
            scp_minmax=list(self._scp_minmax()),
            sorder_minmax=list(self.sorder_minmax),
            sseg_minlength=self.sseg_minlength,
            mcmc_samples=self.mcmc_samples,
            mcmc_burbin=self.mcmc_burnin,
            mcmc_chains=self.mcmc_chains,
            mcmc_thin=self.mcmc_thin,
            mcmc_seed=self.mcmc_seed,
            quiet=True,
        )

    def _raised(self, minmax, used: bool) -> tuple[int, int]:
        """``(min, max)`` number of change points, the max raised to ``max_cps``.

        Rbeast returns at most ``max`` change points per component and
        channel, so a larger ``max_cps`` could never be reached. Only for a
        component whose change points are returned (``used``).
        """
        low, high = int(minmax[0]), int(minmax[1])
        if used and self.max_cps is not None and self.max_cps > high:
            high = self.max_cps
        return low, high

    def _tcp_minmax(self) -> tuple[int, int]:
        return self._raised(self.tcp_minmax, self.component in ("trend", "both"))

    def _scp_minmax(self) -> tuple[int, int]:
        used = self.season != "none" and self.component in ("season", "both")
        return self._raised(self.scp_minmax, used)

    def _run_beast_univariate(self, signal: np.ndarray) -> np.ndarray:
        """Run BEAST on a univariate signal and extract change points."""
        if len(signal) < _MIN_RBEAST_VALUES:
            return np.empty(0, dtype=np.int64)
        rb = _import_rbeast()
        kwargs = self._get_mcmc_kwargs()

        o = rb.beast(signal, **kwargs)
        if _all_nan_output(o):
            _warn_sse_rerun()
            o = rb.beast(signal, **kwargs, cputype="sse")

        return self._postprocess(o, len(signal))

    def _run_beast_multivariate(self, data: np.ndarray) -> np.ndarray:
        """Run beast123 on each channel and merge their change points."""
        n_samples, n_channels = data.shape
        if n_samples < _MIN_RBEAST_VALUES:
            return np.empty(0, dtype=np.int64)
        rb = _import_rbeast()

        md = rb.args()
        md.whichDimIsTime = 1  # rows = time
        md.startTime = 0.0  # CP times / deltat are indices
        md.season = self.season
        md.period = self.period
        md.detrend = False
        md.deltat = self.deltat

        # beast123 reads the prior from an object: fill the fields that
        # Rbeast.beast fills from the same arguments (seasonal ones only with
        # a seasonal component), so that "native" honours them too.
        prior = rb.args()
        prior.trendMinKnotNum, prior.trendMaxKnotNum = self._tcp_minmax()
        prior.trendMinOrder = self.torder_minmax[0]
        prior.trendMaxOrder = self.torder_minmax[1]
        prior.trendMinSepDist = self.tseg_minlength
        if self.season != "none":
            prior.seasonMinKnotNum, prior.seasonMaxKnotNum = self._scp_minmax()
            prior.seasonMinOrder = self.sorder_minmax[0]
            prior.seasonMaxOrder = self.sorder_minmax[1]
            prior.seasonMinSepDist = self.sseg_minlength

        mcmc = rb.args()
        mcmc.samples = self.mcmc_samples
        mcmc.burnin = self.mcmc_burnin
        mcmc.chains = self.mcmc_chains
        mcmc.thin = self.mcmc_thin
        mcmc.seed = self.mcmc_seed

        extra = rb.args()
        extra.quiet = True

        def run_each_channel() -> list:
            # One series per call: a 2D array would go to threads pinned to
            # CPUs 0, 1, ..., which can fail (see the Notes of the class).
            return [
                rb.beast123(
                    np.ascontiguousarray(data[:, ch]),
                    metadata=md,
                    prior=prior,
                    mcmc=mcmc,
                    extra=extra,
                )
                for ch in range(n_channels)
            ]

        outputs = run_each_channel()
        if all(_all_nan_output(o) for o in outputs):
            _warn_sse_rerun()
            extra.cputype = "sse"
            outputs = run_each_channel()

        # Candidates of each channel (and component), with their cpPr.
        per_channel = [
            [
                (cp, pr)
                for cp, pr in self._extract_cps_channel(o, ch)
                if 0 < cp < n_samples
            ]
            for ch, o in enumerate(outputs)
        ]
        # A time step found on several channels (or components) keeps its
        # largest probability.
        probs: dict[int, float] = {}
        channels: dict[int, set[int]] = {}
        for ch, cands in enumerate(per_channel):
            for cp, prob in cands:
                probs[cp] = max(prob, probs.get(cp, prob))
                channels.setdefault(cp, set()).add(ch)

        # Rbeast keeps the change points of one channel at least `separation`
        # apart: closer candidates of different channels are taken as the
        # same change, as are candidates within `tolerance` of each other.
        separation = self._native_separation(n_samples)
        tolerance = self._tolerance_steps(n_samples)

        def same_change(cp: int, other: int) -> bool:
            gap = abs(cp - other)
            return gap <= tolerance or (
                gap < separation and channels[cp].isdisjoint(channels[other])
            )

        # Support: the cpPr of the change at `cp` summed over the channels,
        # each giving its most probable candidate at that step, closer than
        # `separation` or within `tolerance`.
        steps = np.array(sorted(probs), dtype=np.int64)
        support = np.zeros(len(steps))
        for cands in per_channel:
            if not cands:
                continue
            cps = np.array([cp for cp, _ in cands], dtype=np.int64)
            prs = np.array([prob for _, prob in cands], dtype=float)
            gap = np.abs(steps[:, None] - cps[None, :])
            near = (gap < separation) | (gap <= tolerance)
            support += np.where(near, prs[None, :], 0.0).max(axis=1)
        supports = dict(zip(steps.tolist(), support.tolist(), strict=True))

        kept: list[int] = []
        # By decreasing support, then probability, ties by position.
        for cp in sorted(probs, key=lambda cp: (-supports[cp], -probs[cp], cp)):
            if not any(same_change(cp, other) for other in kept):
                kept.append(cp)
        if self.max_cps is not None:
            count = self.max_cps
        else:
            # As many as a channel that changes has (the median over the
            # channels with candidates): the union of the channels grows with
            # their number.
            counts = [len(cands) for cands in per_channel if cands]
            count = int(np.floor(np.median(counts) + 0.5)) if counts else 0
        kept = kept[:count]

        return np.array(sorted(kept), dtype=np.int64)

    def _native_separation(self, n_samples: int) -> int:
        """Rbeast's minimum distance between two trend CPs of one channel.

        As Rbeast 0.1.25 sets it (``rbeast_beastv2_io_in_args.c``):
        ``tseg_minlength`` or, if None, 3 time steps (half the period, in time
        steps, with a seasonal component), at least the largest trend order
        plus one and at most ``n_samples // 2 - 1``.
        """
        separation = self.tseg_minlength
        if separation is None:
            separation = 3
            period = self.period / self.deltat
            # NaN: Rbeast estimates the period; below 2: it fits no season.
            if self.season != "none" and period >= 2:
                separation = int(period) // 2
        separation = max(separation, max(self.torder_minmax) + 1)
        return min(separation, n_samples // 2 - 1)

    def _tolerance_steps(self, n_samples: int) -> int:
        """``tolerance`` in time steps (a float below 1 is a fraction of n)."""
        if isinstance(self.tolerance, float) and self.tolerance < 1.0:
            return int(self.tolerance * n_samples)
        return int(self.tolerance)

    def _extract_cps_channel(self, o, ch: int) -> list[tuple[int, float]]:
        """Extract ``(cp, cpPr)`` pairs from the beast123 output of channel ``ch``."""
        cps: list[tuple[int, float]] = []

        for comp_name in ("trend", "season"):
            if self.component not in (comp_name, "both"):
                continue
            comp = getattr(o, comp_name, None)
            if comp is None:
                continue

            cp_arr = getattr(comp, "cp", None)
            cp_pr = getattr(comp, "cpPr", None)
            ncp_med = getattr(comp, "ncp_median", None)

            if cp_arr is None:
                continue

            cp_col = np.asarray(cp_arr).ravel()
            pr_col = None if cp_pr is None else np.asarray(cp_pr).ravel()

            n_expected = _median_ncp(ncp_med, 0, f"{comp_name}, channel {ch}")

            comp_cps: list[tuple[int, float]] = []
            for i, cp in enumerate(cp_col):
                if np.isnan(cp):
                    break
                prob = pr_col[i] if pr_col is not None and i < len(pr_col) else 1.0
                if float(prob) < self.cp_prob_threshold:
                    continue
                comp_cps.append((int(round(float(cp) / self.deltat)), float(prob)))

            if n_expected is not None and len(comp_cps) > n_expected:
                comp_cps = comp_cps[:n_expected]

            cps.extend(comp_cps)

        return cps

    def _postprocess(self, o, n: int) -> np.ndarray:
        """Extract, filter and cap change points from a beast() output."""
        cps = self._extract_cps(o)

        if cps.size == 0:
            return np.empty(0, dtype=np.int64)

        cps = cps[(cps > 0) & (cps < n)]
        cps = np.unique(cps)

        if self.max_cps is not None and len(cps) > self.max_cps:
            probs = self._extract_cp_probs(o)
            if probs is not None and len(probs) >= len(cps):
                cp_prob_pairs = [
                    (cp, probs[cp] if cp < len(probs) else 0.0) for cp in cps
                ]
                cp_prob_pairs.sort(key=lambda x: x[1], reverse=True)
                cps = np.sort(
                    np.array(
                        [p[0] for p in cp_prob_pairs[: self.max_cps]],
                        dtype=np.int64,
                    )
                )
            else:
                cps = cps[: self.max_cps]

        return cps.astype(np.int64)

    def _extract_cps(self, o) -> np.ndarray:
        """Extract change point indices from a BEAST output object."""
        all_cps = []

        if self.component in ("trend", "both") and hasattr(o, "trend"):
            trend_cps = self._cps_from_component(o.trend, "trend")
            all_cps.extend(trend_cps)

        if self.component in ("season", "both") and hasattr(o, "season"):
            season_cps = self._cps_from_component(o.season, "season")
            all_cps.extend(season_cps)

        if not all_cps:
            return np.empty(0, dtype=np.int64)

        return np.unique(np.array(all_cps, dtype=np.int64))

    def _cps_from_component(self, component, name: str = "trend") -> list[int]:
        """Extract filtered CPs from a single BEAST component (trend/season).

        Uses the ranked ``cp`` / ``cpPr`` arrays from BEAST's output. These
        list the most likely change-point locations sorted by descending
        probability, which is more reliable than thresholding the dense
        ``cpOccPr`` curve (which can be diffuse around the true location).
        The number of CPs is guided by ``ncp_median``; ``name`` (``"trend"``
        or ``"season"``) only labels the warning raised when it is NaN.
        """
        cps: list[int] = []

        # Strategy 1: use cp + cpPr (ranked list)
        if hasattr(component, "cp") and component.cp is not None:
            cp_arr = np.asarray(component.cp).ravel()
            cp_pr = None
            if hasattr(component, "cpPr") and component.cpPr is not None:
                cp_pr = np.asarray(component.cpPr).ravel()

            # Use ncp_median as the expected number of CPs
            n_expected = _median_ncp(getattr(component, "ncp_median", None), 0, name)

            for i, cp in enumerate(cp_arr):
                if np.isnan(cp):
                    break
                prob = cp_pr[i] if cp_pr is not None and i < len(cp_pr) else 1.0
                if float(prob) < self.cp_prob_threshold:
                    continue
                cps.append(int(round(float(cp) / self.deltat)))

            # If we have an expected count, keep at most that many
            if n_expected is not None and len(cps) > n_expected:
                cps = cps[:n_expected]

            return cps

        # Strategy 2: fallback to cpOccPr (dense probability curve)
        if hasattr(component, "cpOccPr") and component.cpOccPr is not None:
            prob = np.asarray(component.cpOccPr).ravel()
            indices = np.where(prob >= self.cp_prob_threshold)[0]

            if len(indices) > 0:
                # Group consecutive indices: pick the local peak in each run
                groups = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
                for group in groups:
                    peak = group[np.argmax(prob[group])]
                    cps.append(int(peak))

        return cps

    def _extract_cp_probs(self, o) -> np.ndarray | None:
        """Return the ``cpOccPr`` curve of the returned component, if any.

        ``cpOccPr[i]`` is Rbeast's probability that the component changes at
        time step ``i``. With ``component="both"``, the larger of the trend
        and seasonal curves at each step (NaN ignored), so that a seasonal
        candidate is not ranked by the trend's curve alone.
        """
        names = ("trend", "season") if self.component == "both" else (self.component,)
        curves = [
            np.asarray(getattr(o, name).cpOccPr).ravel()
            for name in names
            if getattr(getattr(o, name, None), "cpOccPr", None) is not None
        ]
        if curves:
            return np.fmax.reduce(curves)
        return None
