"""Plotting utilities for the interactive demo."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_BASE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _adjust_color_hex(hex_color: str, factor: float) -> str:
    """Lighten the provided hex colour by ``factor`` in [0,1]."""

    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def build_segment_color_map(n_segments: int, n_dims: int) -> list[list[str]]:
    """Return a [segment][dimension] colour mapping."""

    colors = []
    for idx in range(n_segments):
        base = _BASE_COLORS[idx % len(_BASE_COLORS)]
        segment_colors = []
        for dim in range(n_dims):
            factor = 0.15 * dim
            segment_colors.append(_adjust_color_hex(base, factor))
        colors.append(segment_colors)
    return colors


def _iter_segments(states: Sequence[int]) -> Iterable[tuple[int, int, int]]:
    start = 0
    current = states[0]
    for idx in range(1, len(states)):
        if states[idx] != current:
            yield start, idx, current
            start = idx
            current = states[idx]
    yield start, len(states), current


def make_annotation_figure(
    signal: np.ndarray,
    change_points: Sequence[int],
    states: Sequence[int],
    title: str,
    color_map: list[list[str]],
) -> go.Figure:
    """Build a Plotly figure highlighting segments and change points."""

    n_samples, n_dims = signal.shape
    fig = go.Figure()

    for start, stop, state in _iter_segments(states):
        for dim in range(n_dims):
            color = color_map[state % len(color_map)][dim % len(color_map[state % len(color_map)])]
            fig.add_trace(
                go.Scatter(
                    x=np.arange(start, stop),
                    y=signal[start:stop, dim],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"State {state} · Dim {dim}",
                    showlegend=False,
                )
            )

    for cp in change_points:
        fig.add_vline(x=cp, line=dict(color="#333333", width=1, dash="dash"))

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_white",
        height=320 + 40 * n_dims,
    )
    return fig


def make_comparison_figure(
    signal: np.ndarray,
    truth_states: Sequence[int],
    truth_cps: Sequence[int],
    pred_states: Sequence[int] | None,
    pred_cps: Sequence[int] | None,
    color_map: list[list[str]],
    *,
    fallback_states: Sequence[int] | None = None,
) -> go.Figure:
    """Two-row figure contrasting ground-truth and predictions."""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    truth_fig = make_annotation_figure(signal, truth_cps, truth_states, "Ground truth", color_map)
    for trace in truth_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for shape in truth_fig.layout.shapes or []:
        fig.add_shape(shape, row=1, col=1)

    if pred_states is not None:
        pred_fig = make_annotation_figure(signal, pred_cps or [], pred_states, "Prediction", color_map)
        for trace in pred_fig.data:
            fig.add_trace(trace, row=2, col=1)
        for shape in pred_fig.layout.shapes or []:
            fig.add_shape(shape, row=2, col=1)
    elif fallback_states is not None:
        pred_fig = make_annotation_figure(signal, pred_cps or [], fallback_states, "Prediction", color_map)
        for trace in pred_fig.data:
            fig.add_trace(trace, row=2, col=1)
        for shape in pred_fig.layout.shapes or []:
            fig.add_shape(shape, row=2, col=1)
    else:
        fig.add_trace(
            go.Scatter(
                x=np.arange(signal.shape[0]),
                y=signal[:, 0],
                mode="lines",
                name="Signal",
            ),
            row=2,
            col=1,
        )
        if pred_cps:
            for cp in pred_cps:
                fig.add_vline(x=cp, line=dict(color="#333333", width=1, dash="dash"), row=2, col=1)

    fig.update_layout(height=650, template="plotly_white")
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    return fig
