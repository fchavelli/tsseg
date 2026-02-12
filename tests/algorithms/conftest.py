"""Pytest configuration and shared fixtures for the algorithm test suite.

This module auto-discovers **every** public detector listed in
``tsseg.algorithms.__all__`` and exposes them as parametrised fixtures so that
new algorithms are tested automatically — no manual registration required.

Algorithm-specific overrides (custom constructor kwargs, optional dependencies,
etc.) are declared in :data:`ALGORITHM_OVERRIDES`.  Anything *not* listed there
falls back to sensible defaults (default constructor, no special dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np
import pytest

import tsseg.algorithms as _algorithms_module
from tsseg.algorithms.base import BaseSegmenter

# ======================================================================
# Per-algorithm overrides
# ======================================================================


@dataclass(frozen=True)
class AlgorithmOverride:
    """Declare any non-default behaviour for a specific detector.

    Attributes
    ----------
    init_kwargs : dict
        Extra keyword arguments passed to the constructor.
        ``None`` means "use the default constructor with no args".
    dependencies : tuple[str, ...]
        Optional third-party packages.  The algorithm is skipped when any
        of these is missing from the current environment.
    semi_supervised : bool
        If ``True``, the algorithm requires supervision info (``y``, and
        sometimes ``n_segments`` / ``n_states``) to be set before predict.
    skip_reason : str | None
        If set, the algorithm is unconditionally skipped with this message.
    """

    init_kwargs: dict[str, Any] | None = None
    dependencies: tuple[str, ...] = ()
    semi_supervised: bool = False
    skip_reason: str | None = None


# Only algorithms that *cannot* be instantiated with a bare ``Cls()`` need
# an entry here.  Everything else is discovered and tested automatically.
ALGORITHM_OVERRIDES: dict[str, AlgorithmOverride] = {
    # --- Semi-supervised (need n_cps / n_segments via y) ---------------
    "AutoPlaitDetector": AlgorithmOverride(
        init_kwargs={"n_cps": None},
        semi_supervised=True,
    ),
    "DynpDetector": AlgorithmOverride(semi_supervised=True),
    "HidalgoDetector": AlgorithmOverride(semi_supervised=True),
    "VSAXDetector": AlgorithmOverride(semi_supervised=True),
    # --- Need an explicit stopping criterion --------------------------
    "WindowDetector": AlgorithmOverride(init_kwargs={"n_cps": 2}),
    "EspressoDetector": AlgorithmOverride(
        init_kwargs={"n_segments": 3, "window_size": 10},
    ),
    # --- Optional heavy dependencies ----------------------------------
    "E2USDDetector": AlgorithmOverride(dependencies=("torch",)),
    "FLUSSDetector": AlgorithmOverride(dependencies=("stumpy",)),
    "ProphetDetector": AlgorithmOverride(dependencies=("prophet",)),
    "Time2StateDetector": AlgorithmOverride(dependencies=("torch",)),
    "TireDetector": AlgorithmOverride(dependencies=("torch",)),
    "TGLADDetector": AlgorithmOverride(dependencies=("torch", "networkx")),
    "TSCP2Detector": AlgorithmOverride(dependencies=("tensorflow",)),
    "VQTSSDetector": AlgorithmOverride(
        dependencies=("torch",),
        init_kwargs={
            "epochs": 2,
            "batch_size": 16,
            "window_size": 20,
            "num_embeddings": 10,
        },
    ),
    # --- Tirex detectors (need torch + tirex runtime) -----------------
    "TirexHiddenCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexCosineCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexL2CPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexMMDCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexEnergyCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexDerivativeCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexGateRatioCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexForgetDropCPD": AlgorithmOverride(dependencies=("torch", "moment")),
    "TirexForecastErrorCPD": AlgorithmOverride(dependencies=("torch", "moment")),
}

_DEFAULT = AlgorithmOverride()


# ======================================================================
# Helpers
# ======================================================================


def _override_for(name: str) -> AlgorithmOverride:
    return ALGORITHM_OVERRIDES.get(name, _DEFAULT)


def _dependency_missing(dep: str) -> bool:
    try:
        return find_spec(dep) is None
    except ModuleNotFoundError:
        return True


def make_instance(cls, override: AlgorithmOverride | None = None):
    """Instantiate a detector, respecting override kwargs."""
    if override is None:
        override = _DEFAULT
    kwargs = override.init_kwargs or {}
    return cls(**kwargs)


def apply_supervision(instance, y: np.ndarray) -> None:
    """Inject supervision info for semi-supervised algorithms.

    Sets ``n_segments``, ``n_states``, etc. on the instance based on ``y``,
    mimicking what a benchmark harness would do.
    """
    cps = np.flatnonzero(np.diff(y) != 0) + 1
    n_cps = len(cps)
    n_segments = n_cps + 1
    n_states = len(np.unique(y))

    attr_map = {
        # change-point count aliases
        "n_cps": n_cps,
        "n_change_points": n_cps,
        "n_bkps": n_cps,
        "n_breakpoints": n_cps,
        "N": n_cps,
        # segment count aliases
        "n_segments": n_segments,
        # state count aliases
        "n_states": n_states,
        "n_regimes": n_states,
        "n_clusters": n_states,
        "n_components": n_states,
        "K": n_states,
    }
    for attr, value in attr_map.items():
        if hasattr(instance, attr):
            setattr(instance, attr, value)


# ======================================================================
# Session-scoped synthetic data
# ======================================================================


def _segmented_signal(
    rng: np.random.Generator,
    n_samples: int,
    change_points: np.ndarray,
    means: list[np.ndarray],
    scales: list[np.ndarray],
) -> np.ndarray:
    """Generate a piecewise-Gaussian signal with known change points."""
    segments, start = [], 0
    for end, mu, sigma in zip(
        list(change_points) + [n_samples], means, scales
    ):
        seg = rng.normal(loc=mu, scale=sigma, size=(end - start, mu.shape[0]))
        segments.append(seg)
        start = end
    return np.concatenate(segments, axis=0).astype(np.float64)


@pytest.fixture(scope="session")
def synthetic_data():
    """Three-segment synthetic series (univariate + multivariate).

    Change points at index 300 and 700 in a length-1000 signal with well-
    separated segment means so that virtually any algorithm can detect them.
    """
    rng = np.random.default_rng(42)
    n = 1000
    cps = np.array([300, 700])
    labels = np.zeros(n, dtype=int)
    labels[cps[0] : cps[1]] = 1
    labels[cps[1] :] = 2

    uni = _segmented_signal(
        rng,
        n,
        cps,
        means=[np.array([-0.8]), np.array([0.5]), np.array([-0.1])],
        scales=[np.array([0.2]), np.array([0.25]), np.array([0.1])],
    )
    multi = _segmented_signal(
        rng,
        n,
        cps,
        means=[
            np.array([-0.8, 0.4]),
            np.array([0.5, -0.3]),
            np.array([-0.1, 0.6]),
        ],
        scales=[
            np.array([0.2, 0.18]),
            np.array([0.25, 0.22]),
            np.array([0.22, 0.2]),
        ],
    )
    return {
        "univariate": {"X": uni, "y": labels.copy(), "change_points": cps.copy()},
        "multivariate": {"X": multi, "y": labels.copy(), "change_points": cps.copy()},
        "n_samples": n,
    }


# ======================================================================
# Core parametrised fixture — one test-id per algorithm
# ======================================================================


@pytest.fixture(params=_algorithms_module.__all__)
def algorithm(request):
    """Yield ``(name, cls, override, instance)`` for every registered algorithm.

    Algorithms whose optional dependencies are missing are skipped
    automatically.
    """
    name: str = request.param
    cls = getattr(_algorithms_module, name)
    ovr = _override_for(name)

    if ovr.skip_reason:
        pytest.skip(ovr.skip_reason)
    for dep in ovr.dependencies:
        if _dependency_missing(dep):
            pytest.skip(f"Missing optional dependency '{dep}' for {name}")

    instance = make_instance(cls, ovr)
    return name, cls, ovr, instance
