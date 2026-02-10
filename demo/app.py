"""Streamlit app exposing synthetic data generation and segmentation demos."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:  # pragma: no cover - streamlit is an optional dependency for the demo
    import streamlit as st  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The demo module requires the optional dependency 'streamlit'. "
        "Install it with `pip install streamlit`."
    ) from exc

try:
    from .algorithms import DetectorInfo, get_constructor_signature, get_detectors_by_type, instantiate_detector
    from .data import ComplexityLevel, SyntheticSeries, generate_synthetic_series
    from .plotting import build_segment_color_map, make_annotation_figure, make_comparison_figure
    from .utils import (
        align_states_via_hungarian,
        change_points_to_states,
        normalize_zscore,
        parse_user_value,
        states_to_change_points,
    )
except ImportError:  # pragma: no cover - support script execution via streamlit run demo/app.py
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from demo.algorithms import DetectorInfo, get_constructor_signature, get_detectors_by_type, instantiate_detector  # type: ignore[import-not-found]
    from demo.data import ComplexityLevel, SyntheticSeries, generate_synthetic_series  # type: ignore[import-not-found]
    from demo.plotting import build_segment_color_map, make_annotation_figure, make_comparison_figure  # type: ignore[import-not-found]
    from demo.utils import (  # type: ignore[import-not-found]
        align_states_via_hungarian,
        change_points_to_states,
        normalize_zscore,
        parse_user_value,
        states_to_change_points,
    )

_DETECTOR_TYPES = {
    "Change point detection": "change_point_detection",
    "State detection": "state_detection",
}


@st.cache_data(show_spinner=False)
def _generate_series_cached(
    length: int,
    n_segments: int,
    n_dims: int,
    complexity: ComplexityLevel,
    noise_level: float,
    seed: int | None,
) -> SyntheticSeries:
    return generate_synthetic_series(
        length=length,
        n_segments=n_segments,
        n_dims=n_dims,
        complexity=complexity,
        noise_level=noise_level,
        random_state=seed,
    )


def _render_param_controls(detector: DetectorInfo) -> Dict[str, Any]:
    st.markdown("### Algorithm parameters")
    signature = get_constructor_signature(detector.cls)
    params: Dict[str, Any] = {}
    if not signature.parameters:
        st.info("This detector does not expose configurable parameters.")
        return params

    for param in signature.parameters.values():
        default = param.default if param.default is not inspect._empty else None
        label = f"{param.name}"
        key = f"param_{detector.name}_{param.name}"
        annotation = param.annotation if param.annotation is not inspect._empty else None

        if isinstance(default, bool):
            params[param.name] = st.checkbox(label, value=default, key=key)
        elif isinstance(default, int):
            lower = 0 if default >= 0 else int(default * 3)
            upper = default * 3 + 10 if default >= 0 else -default * 3 + 10
            upper = max(upper, lower + 5)
            params[param.name] = st.slider(label, min_value=lower, max_value=upper, value=default, step=1, key=key)
        elif isinstance(default, float):
            span = max(abs(default), 1.0)
            lower = float(-3 * span)
            upper = float(3 * span)
            params[param.name] = st.slider(label, min_value=lower, max_value=upper, value=float(default), step=span / 20, key=key)
        elif default is None:
            raw = st.text_input(label, value="None", key=key)
            params[param.name] = None if raw.strip().lower() in {"none", ""} else parse_user_value(raw, None)
        elif annotation in (list[int], tuple[int, ...], list[float], tuple[float, ...]):  # type: ignore[attr-defined]
            raw = st.text_input(label, value=str(default), key=key)
            params[param.name] = parse_user_value(raw, default)
        else:
            raw = st.text_input(label, value=str(default), key=key)
            params[param.name] = parse_user_value(raw, default)

    return params


def _format_detector_label(info: DetectorInfo) -> str:
    extras = info.tags.get("python_dependencies") or []
    suffix = ""
    if extras:
        suffix = " (requires: " + ", ".join(extras) + ")"
    return f"{info.name}{suffix}"


def _handler_run_algorithm(detector: DetectorInfo, params: Dict[str, Any], series: SyntheticSeries):
    instance = instantiate_detector(detector.cls, params)
    signal = normalize_zscore(series.signal)
    try:
        if hasattr(instance, "fit_predict"):
            output = instance.fit_predict(signal, axis=0)
        else:
            instance.fit(signal, axis=0)
            if hasattr(instance, "predict"):
                output = instance.predict(signal, axis=0)
            elif hasattr(instance, "transform"):
                output = instance.transform(signal)
            else:
                raise RuntimeError("Detector does not expose predict/transform methods")
    except Exception as exc:  # pragma: no cover - surfaced in UI
        raise RuntimeError(f"Algorithm execution failed: {exc}") from exc
    return output


def _interpret_predictions(output: Any, length: int, expect_states: bool):
    pred_states = None
    pred_cps = None
    raw_state_series = None

    if output is None:
        return pred_states, pred_cps

    if isinstance(output, dict) and "change_points" in output:
        pred_cps = [int(x) for x in output["change_points"]]
    elif isinstance(output, (list, tuple)):
        if output and isinstance(output[0], (list, tuple, np.ndarray)):
            primary = output[0]
        else:
            primary = output
        arr = np.asarray(primary)
        if arr.ndim == 1 and arr.size == length:
            raw_state_series = arr.astype(int)
        else:
            pred_cps = [int(x) for x in arr]
    else:
        arr = np.asarray(output)
        if arr.ndim == 1 and arr.size == length:
            raw_state_series = arr.astype(int)
        else:
            pred_cps = [int(x) for x in np.ravel(arr)]

    if expect_states:
        pred_states = raw_state_series
        if pred_states is None and pred_cps is not None:
            pred_states = change_points_to_states(pred_cps, length)
        if pred_states is not None and pred_cps is None:
            pred_cps = states_to_change_points(pred_states)
        return pred_states, pred_cps

    # Change-point detection only: ensure we only expose breakpoints
    if pred_cps is None and raw_state_series is not None:
        pred_cps = states_to_change_points(raw_state_series)
    return None, pred_cps


def main() -> None:
    st.set_page_config(page_title="tsseg demo", layout="wide")
    st.title("Interactive segmentation demo")

    with st.sidebar:
        st.header("1. Synthetic data")
        length = st.slider("Length", min_value=200, max_value=5000, value=1000, step=50)
        n_segments = st.slider("Segments", min_value=1, max_value=10, value=3, step=1)
        n_dims = st.slider("Dimensions", min_value=1, max_value=5, value=1, step=1)
        complexity = st.selectbox("Complexity", options=list(ComplexityLevel.__args__), index=1)  # type: ignore[attr-defined]
        noise_level = st.slider("Noise level", min_value=0.0, max_value=1.5, value=0.5, step=0.05)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

        st.header("2. Algorithm")
        detector_label = st.radio("Task", list(_DETECTOR_TYPES.keys()))
        detector_type = _DETECTOR_TYPES[detector_label]
        expects_states = detector_type == "state_detection"
        available = get_detectors_by_type(detector_type)
        if not available:
            st.error("No detectors available for the selected task.")
            st.stop()
        sorted_names = sorted(available)
        selected_name = st.selectbox("Detector", options=sorted_names, format_func=lambda name: _format_detector_label(available[name]))
        detector_info = available[selected_name]

        params = _render_param_controls(detector_info)

    series = _generate_series_cached(length, n_segments, n_dims, complexity, noise_level, int(seed))
    n_states = int(series.states.max() + 1)
    color_map = build_segment_color_map(n_states, n_dims)

    st.subheader("Ground truth")
    ground_fig = make_annotation_figure(
        signal=series.signal,
        change_points=series.change_points,
        states=series.states,
        title="Ground truth",
        color_map=color_map,
    )
    st.plotly_chart(ground_fig, use_container_width=True)

    if st.button("Run algorithm", type="primary"):
        with st.spinner("Applying detector..."):
            try:
                output = _handler_run_algorithm(detector_info, params, series)
                pred_states, pred_cps = _interpret_predictions(output, length, expects_states)
            except Exception as exc:  # pragma: no cover - show error in UI
                st.error(str(exc))
                return

        if expects_states and pred_states is not None:
            aligned_states = align_states_via_hungarian(series.states, pred_states)
        else:
            aligned_states = None

        if expects_states:
            pred_cps = pred_cps or (states_to_change_points(aligned_states) if aligned_states is not None else [])
        else:
            pred_cps = pred_cps or []

        result_fig = make_comparison_figure(
            signal=series.signal,
            truth_states=series.states,
            truth_cps=series.change_points,
            pred_states=aligned_states,
            pred_cps=pred_cps,
            color_map=color_map,
            fallback_states=None if expects_states else series.states,
        )
        st.subheader("Prediction")
        st.plotly_chart(result_fig, use_container_width=True)

        st.markdown("**Predicted change points**: %s" % pred_cps)


if __name__ == "__main__":  # pragma: no cover
    main()
