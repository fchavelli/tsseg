from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

ComplexityLevel = Literal["simple", "medium", "complex"]


@dataclass
class SyntheticSeries:
    signal: np.ndarray
    change_points: list[int]
    change_points_one_hot: np.ndarray
    states: np.ndarray


def _generate_boundaries(length: int, n_segments: int, rng: np.random.Generator) -> list[int]:
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    if length <= n_segments:
        raise ValueError("length must exceed number of segments")

    # Dirichlet sample ensures positive segment lengths that sum to length.
    proportions = rng.dirichlet(alpha=np.ones(n_segments))
    segment_lengths = np.maximum(5, (proportions * length).astype(int))
    delta = length - int(segment_lengths.sum())
    segment_lengths[0] += delta  # distribute rounding difference to first segment

    boundaries: list[int] = []
    cursor = 0
    for seg_len in segment_lengths[:-1]:
        cursor += int(seg_len)
        boundaries.append(min(cursor, length - 1))
    return [cp for cp in boundaries if cp < length]


def _sine_segment(
    start: int,
    stop: int,
    n_dims: int,
    mean_offset: float,
    amplitude: float,
    freq: float,
    phase: float,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.linspace(0, 1, stop - start, endpoint=False)
    base = amplitude * np.sin(2 * math.pi * (freq * t + phase)) + mean_offset
    noise_scale = (0.05 * amplitude if amplitude else 0.05) * noise_level
    if noise_scale <= 0:
        noise = np.zeros((stop - start, n_dims))
    else:
        noise = rng.normal(scale=noise_scale, size=(stop - start, n_dims))
    return np.tile(base[:, None], (1, n_dims)) + noise


def _complex_segment(
    start: int,
    stop: int,
    n_dims: int,
    trend: float,
    curvature: float,
    freq: float,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    t = np.linspace(0, 1, stop - start, endpoint=False)
    base = trend * t + curvature * (t - 0.5) ** 2
    seasonal = np.sin(2 * math.pi * freq * t)
    features = base + 0.6 * seasonal
    out = np.empty((stop - start, n_dims))
    for dim in range(n_dims):
        noise_scale = 0.1 * noise_level
        drift_scale = 0.2 * noise_level
        noise = rng.normal(scale=noise_scale, size=features.shape) if noise_scale > 0 else np.zeros_like(features)
        drift = (rng.normal(scale=drift_scale) * t) if drift_scale > 0 else 0.0
        out[:, dim] = features + drift + noise
    return out


def generate_synthetic_series(
    length: int = 1_000,
    n_segments: int = 3,
    n_dims: int = 1,
    complexity: ComplexityLevel = "medium",
    noise_level: float = 0.5,
    random_state: int | None = None,
) -> SyntheticSeries:
    """Generate a synthetic multivariate time series with labelled segments."""

    if length <= 0:
        raise ValueError("length must be positive")
    if n_dims <= 0:
        raise ValueError("n_dims must be positive")

    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")

    rng = np.random.default_rng(random_state)
    change_points = _generate_boundaries(length=length, n_segments=n_segments, rng=rng)
    states = np.zeros(length, dtype=int)
    signal = np.zeros((length, n_dims), dtype=float)

    boundaries = [0, *change_points, length]

    for idx, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if complexity == "simple":
            level = np.full((n_dims,), float(idx))
            if noise_level > 0:
                level += rng.normal(scale=0.2 * noise_level, size=(n_dims,))
            noise_scale = (0.05 + 0.05 * idx) * noise_level
            noise = (
                rng.normal(scale=noise_scale, size=(stop - start, n_dims))
                if noise_scale > 0
                else np.zeros((stop - start, n_dims))
            )
            segment = level + noise
        elif complexity == "medium":
            amplitude = 1.0 + 0.5 * idx
            mean_offset = idx * 0.5
            freq = 1 + 0.5 * idx
            phase = rng.uniform(0, 1)
            segment = _sine_segment(
                start,
                stop,
                n_dims,
                mean_offset,
                amplitude,
                freq,
                phase,
                noise_level,
                rng,
            )
        elif complexity == "complex":
            trend = rng.normal(loc=0.5 * idx, scale=0.2)
            curvature = rng.uniform(-1.0, 1.0)
            freq = rng.uniform(1.0, 3.0)
            segment = _complex_segment(
                start,
                stop,
                n_dims,
                trend,
                curvature,
                freq,
                noise_level,
                rng,
            )
        else:
            raise ValueError(f"Unknown complexity level: {complexity}")

        signal[start:stop] = segment
        states[start:stop] = idx

    one_hot = np.zeros(length, dtype=int)
    for cp in change_points:
        if 0 <= cp < length:
            one_hot[cp] = 1

    return SyntheticSeries(
        signal=signal,
        change_points=change_points,
        change_points_one_hot=one_hot,
        states=states,
    )
