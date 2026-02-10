from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Dict

import numpy as np
import pytest

import tsseg.algorithms as algorithms_module
from tsseg.algorithms.utils import extract_cps


@dataclass(frozen=True)
class AlgorithmConfig:
    init_with_defaults: bool = True
    init_kwargs: Dict[str, Any] | None = None
    fit_init_kwargs: Dict[str, Any] | None = None
    fit_kwargs: Dict[str, Any] | None = None
    predict_kwargs: Dict[str, Any] | None = None
    semi_supervised: bool = False
    dependencies: tuple[str, ...] = ()
    skip_reason: str | None = None
    skip_fit_predict: bool = False
    skip_detection: bool = False
    detection_tolerance: int = 40


DEFAULT_CONFIG = AlgorithmConfig()


ALGORITHM_CONFIGS: Dict[str, AlgorithmConfig] = {
    # --- Default (no special config) ---
    "AmocDetector": DEFAULT_CONFIG,
    "BOCDDetector": DEFAULT_CONFIG,
    "BottomUpDetector": DEFAULT_CONFIG,
    "ClapDetector": DEFAULT_CONFIG,
    "ClaspDetector": DEFAULT_CONFIG,
    "EAggloDetector": DEFAULT_CONFIG,
    "EspressoDetector": AlgorithmConfig(
        init_with_defaults=False,
        init_kwargs={"n_segments": 3, "window_size": 10},
        skip_detection=True,
        detection_tolerance=40,
    ),
    "GreedyGaussianDetector": DEFAULT_CONFIG,
    "ICIDDetector": DEFAULT_CONFIG,
    "KCPDDetector": DEFAULT_CONFIG,
    "PeltDetector": DEFAULT_CONFIG,
    "PatssDetector": DEFAULT_CONFIG,
    "RandomDetector": DEFAULT_CONFIG,
    "TiccDetector": DEFAULT_CONFIG,
    # --- Multivariate-only ---
    "InformationGainDetector": DEFAULT_CONFIG,
    # --- Semi-supervised (y required or expected) ---
    "AutoPlaitDetector": AlgorithmConfig(
        init_kwargs={"n_cps": None},
        semi_supervised=True,
    ),
    "DynpDetector": AlgorithmConfig(
        semi_supervised=True,
    ),
    "HidalgoDetector": AlgorithmConfig(
        semi_supervised=True,
    ),
    "VSAXDetector": AlgorithmConfig(
        semi_supervised=True,
    ),
    # --- Require a stopping criterion ---
    "WindowDetector": AlgorithmConfig(
        init_with_defaults=False,
        init_kwargs={"n_cps": 2},
    ),
    # --- Optional heavy dependencies ---
    "BinSegDetector": DEFAULT_CONFIG,
    "E2USDDetector": AlgorithmConfig(
        dependencies=("torch",),
    ),
    "FLUSSDetector": AlgorithmConfig(
        dependencies=("stumpy",),
    ),
    "HdpHsmmDetector": AlgorithmConfig(
        dependencies=("pyhsmm",),
    ),
    "HdpHsmmLegacyDetector": AlgorithmConfig(
        dependencies=("pyhsmm",),
    ),
    "ProphetDetector": AlgorithmConfig(
        dependencies=("prophet",),
    ),
    "Time2StateDetector": AlgorithmConfig(
        dependencies=("torch",),
    ),
    "TireDetector": AlgorithmConfig(
        dependencies=("torch",),
    ),
    "TGLADDetector": AlgorithmConfig(
        dependencies=("torch", "networkx"),
    ),
    "TSCP2Detector": AlgorithmConfig(
        dependencies=("tensorflow",),
    ),
    "VQTSSDetector": AlgorithmConfig(
        dependencies=("torch",),
        init_kwargs={"epochs": 2, "batch_size": 16, "window_size": 20, "num_embeddings": 10},
    ),
}


def _build_config(name: str) -> AlgorithmConfig:
    base = DEFAULT_CONFIG
    overrides = ALGORITHM_CONFIGS.get(name)
    if overrides is None:
        return base
    return AlgorithmConfig(
        init_with_defaults=overrides.init_with_defaults,
        init_kwargs=overrides.init_kwargs or base.init_kwargs,
        fit_init_kwargs=overrides.fit_init_kwargs or overrides.init_kwargs or base.fit_init_kwargs,
        fit_kwargs=overrides.fit_kwargs or base.fit_kwargs,
        predict_kwargs=overrides.predict_kwargs or base.predict_kwargs,
        semi_supervised=overrides.semi_supervised or base.semi_supervised,
        dependencies=overrides.dependencies or base.dependencies,
        skip_reason=overrides.skip_reason or base.skip_reason,
        skip_fit_predict=overrides.skip_fit_predict or base.skip_fit_predict,
        skip_detection=overrides.skip_detection or base.skip_detection,
        detection_tolerance=overrides.detection_tolerance or base.detection_tolerance,
    )


def _dependency_missing(dep: str) -> bool:
    try:
        return find_spec(dep) is None
    except ModuleNotFoundError:
        return True


@pytest.fixture(scope="session")
def algorithm_names() -> list[str]:
    return list(algorithms_module.__all__)


@pytest.fixture(params=algorithms_module.__all__)
def algorithm_entry(request):
    name: str = request.param
    cls = getattr(algorithms_module, name)
    config = _build_config(name)

    if config.skip_reason:
        pytest.skip(config.skip_reason)

    for dependency in config.dependencies:
        if _dependency_missing(dependency):
            pytest.skip(f"Missing optional dependency '{dependency}' for {name}")

    return {
        "name": name,
        "cls": cls,
        "config": config,
    }


def _generate_segmented_signal(
    rng: np.random.Generator,
    n_samples: int,
    change_points: np.ndarray,
    segment_means: list[np.ndarray],
    segment_scales: list[np.ndarray],
) -> np.ndarray:
    segments = []
    start = 0
    bounds = list(change_points) + [n_samples]
    for end, mean, scale in zip(bounds, segment_means, segment_scales):
        length = end - start
        mean = np.asarray(mean, dtype=float)
        scale = np.asarray(scale, dtype=float)
        segment = rng.normal(loc=mean, scale=scale, size=(length, mean.shape[0]))
        segments.append(segment)
        start = end

    signal = np.concatenate(segments, axis=0)
    return signal.astype(float)


@pytest.fixture(scope="session")
def synthetic_series():
    rng = np.random.default_rng(1234)
    n_samples = 1000
    change_points = np.array([300, 700])
    labels = np.zeros(n_samples, dtype=int)
    labels[change_points[0] : change_points[1]] = 1
    labels[change_points[1] :] = 2

    univariate_means = [
        np.array([-0.8]),
        np.array([0.5]),
        np.array([-0.1]),
    ]
    univariate_scales = [
        np.array([0.2]),
        np.array([0.25]),
        np.array([0.1]),
    ]
    univariate_signal = _generate_segmented_signal(
        rng, n_samples, change_points, univariate_means, univariate_scales
    )

    multivariate_means = [
        np.array([-0.8, 0.4]),
        np.array([0.5, -0.3]),
        np.array([-0.1, 0.6]),
    ]
    multivariate_scales = [
        np.array([0.2, 0.18]),
        np.array([0.25, 0.22]),
        np.array([0.22, 0.2]),
    ]
    multivariate_signal = _generate_segmented_signal(
        rng, n_samples, change_points, multivariate_means, multivariate_scales
    )

    return {
        "univariate": {
            "X": univariate_signal,
            "y": labels.copy(),
            "change_points": change_points.copy(),
        },
        "multivariate": {
            "X": multivariate_signal,
            "y": labels.copy(),
            "change_points": change_points.copy(),
        },
    }


def _instantiate(cls, config: AlgorithmConfig, *, for_fit: bool = False):
    if for_fit and config.fit_init_kwargs is not None:
        return cls(**config.fit_init_kwargs)
    if config.init_with_defaults:
        try:
            return cls()
        except TypeError:
            kwargs = config.init_kwargs or {}
            return cls(**kwargs)
    kwargs = config.init_kwargs or {}
    return cls(**kwargs)


def _apply_supervision_overrides(instance, y):
    if y is None:
        return

    cps = extract_cps(y)
    n_cps = len(cps)
    n_segments = n_cps + 1
    n_states = len(np.unique(y))

    # Map of attribute name -> value
    overrides = {}
    
    # Change point aliases
    for alias in {
        "n_cps", "n_change_points", "num_change_points", "n_breakpoints",
        "num_breakpoints", "n_cp", "n_changepoints", "k_max", "max_cps",
        "n_bkps", "N"
    }:
        overrides[alias] = n_cps

    # Segment count aliases
    for alias in {
        "n_segments", "num_segments", "segments", "n_seg", "num_seg"
    }:
        overrides[alias] = n_segments

    # State count aliases
    for alias in {
        "n_states", "n_regimes", "num_states", "num_regimes", "n_classes",
        "num_classes", "n_clusters", "num_clusters", "n_max_states",
        "alphabet_size", "K_states", "n_components", "number_of_clusters", "K"
    }:
        overrides[alias] = n_states

    for attr, value in overrides.items():
        if hasattr(instance, attr):
            setattr(instance, attr, value)


def _call_fit_predict(instance, config: AlgorithmConfig, X, y):
    fit_kwargs = {"axis": 0}
    if config.fit_kwargs:
        fit_kwargs.update(config.fit_kwargs)
    predict_kwargs = {"axis": 0}
    if config.predict_kwargs:
        predict_kwargs.update(config.predict_kwargs)

    y_arg = y if config.semi_supervised else None

    if config.semi_supervised and y is not None:
        _apply_supervision_overrides(instance, y)

    try:
        result = instance.fit_predict(X, y_arg, **predict_kwargs)
    except AttributeError:
        instance.fit(X, y_arg, **fit_kwargs)
        result = instance.predict(X, **predict_kwargs)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Missing dependency during fit_predict: {exc}")
    except ImportError as exc:
        pytest.skip(f"Import error during fit_predict: {exc}")
    except NotImplementedError as exc:
        pytest.skip(str(exc))

    return result


def _coerce_result(result_obj, returns_dense: bool) -> tuple[np.ndarray, np.ndarray, object]:
    """Convert algorithm outputs into an array while preserving metadata."""

    if isinstance(result_obj, dict):
        change_points = np.asarray(result_obj.get("change_points", []))
        states = np.asarray(result_obj.get("states", []))
        primary = change_points if returns_dense else states
        return primary, change_points, result_obj

    arr = np.asarray(result_obj)
    return arr, np.asarray([]), result_obj


@pytest.fixture
def algorithm_instance(algorithm_entry):
    info = algorithm_entry
    instance = _instantiate(info["cls"], info["config"], for_fit=False)
    return instance, info["config"]


@pytest.fixture
def fit_predict_result(algorithm_entry, synthetic_series):
    info = algorithm_entry
    config = info["config"]
    if config.skip_fit_predict:
        pytest.skip("fit_predict skipped for this algorithm")

    instance = _instantiate(info["cls"], config, for_fit=False)
    datasets = {}

    if instance.get_tag("capability:univariate"):
        data_uni = synthetic_series["univariate"]
        uni_instance = _instantiate(info["cls"], config, for_fit=True)
        result_uni = _call_fit_predict(uni_instance, config, data_uni["X"], data_uni["y"])
        returns_dense_uni = uni_instance.get_tag("returns_dense")
        primary_uni, cp_pred_uni, raw_uni = _coerce_result(result_uni, returns_dense_uni)
        datasets["univariate"] = {
            "X": data_uni["X"],
            "y": data_uni["y"],
            "change_points": data_uni["change_points"],
            "result": primary_uni,
            "predicted_change_points": cp_pred_uni,
            "raw_result": raw_uni,
        }

    if instance.get_tag("capability:multivariate"):
        data_multi = synthetic_series["multivariate"]
        multi_instance = _instantiate(info["cls"], config, for_fit=True)
        result_multi = _call_fit_predict(
            multi_instance, config, data_multi["X"], data_multi["y"]
        )
        returns_dense_multi = multi_instance.get_tag("returns_dense")
        primary_multi, cp_pred_multi, raw_multi = _coerce_result(result_multi, returns_dense_multi)
        datasets["multivariate"] = {
            "X": data_multi["X"],
            "y": data_multi["y"],
            "change_points": data_multi["change_points"],
            "result": primary_multi,
            "predicted_change_points": cp_pred_multi,
            "raw_result": raw_multi,
        }

    if not datasets:
        pytest.skip("Algorithm does not support provided synthetic datasets")

    return {
        "instance": instance,
        "config": config,
        "results": datasets,
    }
