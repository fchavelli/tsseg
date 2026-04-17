"""BEAST (Bayesian Estimator of Abrupt change, Seasonality, and Trend) detector."""

from __future__ import annotations

import os
import sys

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


def _import_rbeast():
    """Import Rbeast, preferring the vendorized build in ``c/Rbeast``.

    Resolution order:
      1. Vendorized build: ``<package_root>/c/Rbeast/py_src`` (compiled .so).
      2. System-installed ``Rbeast`` (``pip install Rbeast``).

    To build the vendoized version, run ``make`` in ``c/Rbeast/``.
    """
    # 1. Try vendorized build
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _vendor_dir = os.path.normpath(
        os.path.join(
            _pkg_root, os.pardir, os.pardir, os.pardir, "c", "Rbeast", "py_src"
        )
    )
    if os.path.isdir(_vendor_dir):
        if _vendor_dir not in sys.path:
            sys.path.insert(0, _vendor_dir)
        try:
            import Rbeast as rb  # noqa: N813

            return rb
        except ImportError:
            pass

    # 2. System install
    try:
        import Rbeast as rb  # noqa: N813

        return rb
    except ImportError:
        pass

    raise ImportError(
        "BeastDetector requires Rbeast. Either:\n"
        "  • Build the vendorized version: cd c/Rbeast && make\n"
        "  • Or install from PyPI:         pip install Rbeast"
    )


class BeastDetector(BaseSegmenter):
    """Bayesian ensemble change-point detector using BEAST (Rbeast).

    BEAST decomposes a time series into trend + seasonality + noise via
    Bayesian model averaging over a large space of piecewise-linear trend
    models and piecewise-harmonic seasonal models. Change points are inferred
    from the posterior distribution, with each time step receiving a
    probability of being a change point.

    This detector wraps the ``Rbeast`` C library. A vendorized copy of the C
    source is shipped in ``c/Rbeast/`` and can be compiled with ``make``.
    Alternatively, ``pip install Rbeast`` works (requires numpy < 2).

    Parameters
    ----------
    season : str, default="none"
        Seasonal model type. ``"none"`` for trend-only data, ``"harmonic"``
        for data with periodic components.
    tcp_minmax : tuple of int, default=(0, 10)
        Minimum and maximum number of trend change points.
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
    mcmc_seed : int, default=0
        Random seed for reproducibility. ``0`` means no fixed seed.
    cp_prob_threshold : float, default=0.1
        Minimum posterior probability of being a change point for a time
        step to be selected.
    max_cps : int or None, default=None
        Maximum number of change points to return (by decreasing probability).
        ``None`` returns all candidates above ``cp_prob_threshold``.
    component : str, default="trend"
        Which component's change points to return: ``"trend"`` or
        ``"season"`` or ``"both"``.
    multivariate_strategy : str, default="native"
        Strategy for multivariate inputs: ``"native"`` uses ``beast123``
        which handles multi-dimensional data natively; ``"l2"`` reduces
        channels via L2 norm; ``"ensembling"`` runs BEAST per channel and
        aggregates.
    tolerance : int or float, default=0
        Tolerance for aggregating change points (ensembling strategy only).
    axis : int, default=0
        Time axis of the input array.
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
            description="Random seed (0 = no fixed seed).",
        ),
        "cp_prob_threshold": ParamDef(
            constraint=Interval(float, 0, 1, Closed.NEITHER),
            description="Min posterior probability to accept a change point.",
        ),
        "max_cps": ParamDef(
            constraint=Interval(int, 1, None, Closed.LEFT),
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
            description="Tolerance for aggregating CPs in ensembling.",
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
        mcmc_seed: int = 0,
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
            season=self.season,
            deltat=self.deltat,
            period=self.period,
            tcp_minmax=list(self.tcp_minmax),
            torder_minmax=list(self.torder_minmax),
            tseg_minlength=self.tseg_minlength,
            scp_minmax=list(self.scp_minmax),
            sorder_minmax=list(self.sorder_minmax),
            sseg_minlength=self.sseg_minlength,
            mcmc_samples=self.mcmc_samples,
            mcmc_burbin=self.mcmc_burnin,
            mcmc_chains=self.mcmc_chains,
            mcmc_thin=self.mcmc_thin,
            mcmc_seed=self.mcmc_seed,
            quiet=True,
        )

    def _run_beast_univariate(self, signal: np.ndarray) -> np.ndarray:
        """Run BEAST on a univariate signal and extract change points."""
        rb = _import_rbeast()

        o = rb.beast(signal, **self._get_mcmc_kwargs())

        return self._postprocess(o, len(signal))

    def _run_beast_multivariate(self, data: np.ndarray) -> np.ndarray:
        """Run BEAST natively on multivariate data via beast123."""
        rb = _import_rbeast()
        n_samples, n_channels = data.shape

        md = rb.args()
        md.whichDimIsTime = 1  # rows = time
        md.season = self.season
        md.period = self.period
        md.detrend = False
        md.deltat = self.deltat

        mcmc = rb.args()
        mcmc.samples = self.mcmc_samples
        mcmc.burnin = self.mcmc_burnin
        mcmc.chains = self.mcmc_chains
        mcmc.thin = self.mcmc_thin
        mcmc.seed = self.mcmc_seed

        extra = rb.args()
        extra.quiet = True

        o = rb.beast123(data, metadata=md, mcmc=mcmc, extra=extra)

        # beast123 returns a single object with per-channel arrays
        # e.g. o.trend.cp has shape (max_cps, n_channels)
        all_cps: list[int] = []
        for ch in range(n_channels):
            cps_ch = self._extract_cps_channel(o, ch)
            all_cps.extend(cps_ch)

        if not all_cps:
            return np.empty(0, dtype=np.int64)

        cps = np.unique(np.array(all_cps, dtype=np.int64))
        cps = cps[(cps > 0) & (cps < n_samples)]

        if self.max_cps is not None and len(cps) > self.max_cps:
            cps = cps[: self.max_cps]

        return cps.astype(np.int64)

    def _extract_cps_channel(self, o, ch: int) -> list[int]:
        """Extract CPs for a single channel from a beast123 output."""
        cps: list[int] = []

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

            cp_arr = np.asarray(cp_arr)
            # beast123 returns (max_cp_slots, n_channels)
            if cp_arr.ndim == 2:
                cp_col = cp_arr[:, ch]
            else:
                cp_col = cp_arr.ravel()

            pr_col = None
            if cp_pr is not None:
                cp_pr = np.asarray(cp_pr)
                pr_col = cp_pr[:, ch] if cp_pr.ndim == 2 else cp_pr.ravel()

            n_expected = None
            if ncp_med is not None:
                ncp_med = np.asarray(ncp_med).ravel()
                if ch < len(ncp_med):
                    n_expected = int(round(float(ncp_med[ch])))

            comp_cps: list[int] = []
            for i, cp in enumerate(cp_col):
                if np.isnan(cp):
                    break
                prob = pr_col[i] if pr_col is not None and i < len(pr_col) else 1.0
                if float(prob) < self.cp_prob_threshold:
                    continue
                comp_cps.append(int(round(float(cp))))

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
            trend_cps = self._cps_from_component(o.trend)
            all_cps.extend(trend_cps)

        if self.component in ("season", "both") and hasattr(o, "season"):
            season_cps = self._cps_from_component(o.season)
            all_cps.extend(season_cps)

        if not all_cps:
            return np.empty(0, dtype=np.int64)

        return np.unique(np.array(all_cps, dtype=np.int64))

    def _cps_from_component(self, component) -> list[int]:
        """Extract filtered CPs from a single BEAST component (trend/season).

        Uses the ranked ``cp`` / ``cpPr`` arrays from BEAST's output. These
        list the most likely change-point locations sorted by descending
        probability, which is more reliable than thresholding the dense
        ``cpOccPr`` curve (which can be diffuse around the true location).
        The number of CPs is guided by ``ncp_median``.
        """
        cps: list[int] = []

        # Strategy 1: use cp + cpPr (ranked list)
        if hasattr(component, "cp") and component.cp is not None:
            cp_arr = np.asarray(component.cp).ravel()
            cp_pr = None
            if hasattr(component, "cpPr") and component.cpPr is not None:
                cp_pr = np.asarray(component.cpPr).ravel()

            # Use ncp_median as the expected number of CPs
            n_expected = None
            if hasattr(component, "ncp_median") and component.ncp_median is not None:
                val = np.asarray(component.ncp_median).ravel()
                if len(val) > 0:
                    n_expected = int(round(float(val[0])))

            for i, cp in enumerate(cp_arr):
                if np.isnan(cp):
                    break
                prob = cp_pr[i] if cp_pr is not None and i < len(cp_pr) else 1.0
                if float(prob) < self.cp_prob_threshold:
                    continue
                cps.append(int(round(float(cp))))

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
        """Return the trend cpOccPr array if available."""
        if hasattr(o, "trend") and hasattr(o.trend, "cpOccPr"):
            return np.asarray(o.trend.cpOccPr).ravel()
        return None

    # ------------------------------------------------------------------
    # Public API extensions
    # ------------------------------------------------------------------

    def get_fitted_params(self):
        return {
            "season": self.season,
            "tcp_minmax": self.tcp_minmax,
            "scp_minmax": self.scp_minmax,
            "mcmc_samples": self.mcmc_samples,
            "cp_prob_threshold": self.cp_prob_threshold,
            "component": self.component,
        }
