"""Variable-length SAX baseline detector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
from ..base import BaseSegmenter
from .breakpoints import get_breakpoints


@dataclass(frozen=True)
class _SegmentChoice:
    """Helper container storing the best choice for a dynamic-programming step."""

    end: int
    symbol: Tuple[int, ...]
    cost: float


class VSAXDetector(BaseSegmenter):
    """Greedy baseline for state detection using variable-length SAX symbols.

    The detector implements a light-weight segmentation strategy suitable as a
    reference baseline for dense state labelling tasks. Given a list of
    admissible segment lengths, it enumerates symbolic aggregate approXimation
    (SAX) representations and retains the sequence that minimises a simple
    reconstruction error with an additive penalty that discourages excessive
    fragmentation.

    Parameters
    ----------
    axis : int, default=0
        Time axis. ``axis=0`` assumes ``(n_timepoints, n_channels)`` input.
    alphabet_size : int, default=6
        Number of SAX symbols. Values >= 1 are supported.
    paa_segments : int, default=8
        Maximum number of PAA frames per segment. Short segments automatically
        reduce the number of frames so that every frame contains at least one
        sample.
    min_segment_length : int, default=20
        Minimum admissible segment length (in samples).
    max_segment_length : int, default=180
        Maximum admissible segment length. The final segment may be shorter if
        the remaining suffix is smaller than ``min_segment_length``.
    num_lengths : int, default=6
        Number of candidate lengths linearly spaced between ``min`` and ``max``.
        Increasing this value improves flexibility at the cost of runtime.
    penalty : float, default=0.8
        Cost added for every new segment. Use larger values to favour longer
        segments; reduce to obtain more change points.
    zscore : bool, default=True
        Apply per-channel z-normalisation before computing scores.
    random_state : int | None, default=None
        Accepted for API compatibility but unused (the algorithm is deterministic).
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        *,
        axis: int = 0,
        alphabet_size: int = 6,
        paa_segments: int = 8,
        min_segment_length: int = 20,
        max_segment_length: int = 180,
        num_lengths: int = 6,
        penalty: float = 0.8,
        zscore: bool = True,
        random_state: int | None = 0,
    ) -> None:
        if alphabet_size < 1:
            raise ValueError(
                "alphabet_size must be an integer >= 1"
            )
        if min_segment_length < 1:
            raise ValueError("min_segment_length must be >= 1")
        if max_segment_length < min_segment_length:
            raise ValueError("max_segment_length must be >= min_segment_length")
        if num_lengths < 1:
            raise ValueError("num_lengths must be >= 1")
        if paa_segments < 1:
            raise ValueError("paa_segments must be >= 1")
        if penalty < 0:
            raise ValueError("penalty must be non-negative")

        self.alphabet_size = alphabet_size
        self.paa_segments = paa_segments
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.num_lengths = num_lengths
        self.penalty = penalty
        self.zscore = zscore
        self.random_state = random_state

        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # tsseg estimator API
    # ------------------------------------------------------------------
    def _fit(self, X: np.ndarray, y: np.ndarray | None = None, axis: int | None = None):
        X = self._validate_data(X)
        self._breakpoints = get_breakpoints(self.alphabet_size)
        self._n_channels = X.shape[1]
        return self

    def _predict(self, X: np.ndarray, axis: int | None = None) -> np.ndarray:
        X = self._validate_data(X)
        n_samples, _ = X.shape

        if n_samples == 0:
            return np.array([], dtype=int)

        if self.zscore:
            series = self._z_normalise(X)
        else:
            series = X.astype(float, copy=True)

        candidate_lengths = self._candidate_lengths(n_samples)
        if len(candidate_lengths) == 0:
            return np.zeros(n_samples, dtype=int)

        dp_cost = np.full(n_samples + 1, np.inf, dtype=float)
        dp_choice: list[_SegmentChoice | None] = [None] * (n_samples + 1)
        dp_cost[n_samples] = 0.0
        dp_choice[n_samples] = _SegmentChoice(n_samples, tuple(), 0.0)

        # Dynamic programming from the end of the series to the start
        for start in range(n_samples - 1, -1, -1):
            best_choice: _SegmentChoice | None = None
            max_length = n_samples - start
            for length in self._enumerate_lengths(candidate_lengths, max_length):
                end = start + length
                segment = series[start:end]
                seg_cost, symbol = self._segment_cost(symbol_series=segment)
                total_cost = seg_cost + self.penalty + dp_cost[end]
                if best_choice is None or total_cost < best_choice.cost:
                    best_choice = _SegmentChoice(end=end, symbol=symbol, cost=total_cost)

            if best_choice is None:
                # Fallback: consume the rest of the sequence as one segment
                end = n_samples
                seg_cost, symbol = self._segment_cost(symbol_series=series[start:end])
                best_choice = _SegmentChoice(end=end, symbol=symbol, cost=seg_cost + self.penalty)

            dp_cost[start] = best_choice.cost
            dp_choice[start] = best_choice

        # Reconstruct state labels
        labels = np.empty(n_samples, dtype=int)
        symbol_to_state: dict[Tuple[int, ...], int] = {}
        next_state = 0
        idx = 0
        while idx < n_samples:
            choice = dp_choice[idx]
            if choice is None or choice.end <= idx:
                # Safety fallback: assign remaining samples to a new state
                labels[idx:] = next_state
                break
            if choice.symbol not in symbol_to_state:
                symbol_to_state[choice.symbol] = next_state
                next_state += 1
            state = symbol_to_state[choice.symbol]
            labels[idx:choice.end] = state
            idx = choice.end

        return labels

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _validate_data(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(f"Input X must be 1D or 2D; received {X.ndim}D input")

        if self.axis == 1:
            X = X.T
        return X.astype(float, copy=False)

    def _z_normalise(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        return (X - mean) / std

    def _candidate_lengths(self, n_samples: int) -> np.ndarray:
        if n_samples < self.min_segment_length:
            return np.array([n_samples], dtype=int)
        values = np.linspace(
            self.min_segment_length,
            min(self.max_segment_length, n_samples),
            num=self.num_lengths,
            dtype=float,
        )
        lengths = np.unique(values.round().astype(int))
        lengths = lengths[lengths >= self.min_segment_length]
        lengths = lengths[lengths <= n_samples]
        if lengths.size == 0:
            return np.array([n_samples], dtype=int)
        return lengths

    def _enumerate_lengths(
        self, candidate_lengths: Iterable[int], max_length: int
    ) -> Iterable[int]:
        yielded = False
        for length in candidate_lengths:
            if length <= max_length:
                yielded = True
                yield int(length)
        if not yielded:
            yield max_length

    def _segment_cost(self, symbol_series: np.ndarray) -> Tuple[float, Tuple[int, ...]]:
        length = symbol_series.shape[0]
        if length == 0:
            return 0.0, tuple()

        frames = min(self.paa_segments, length)
        frame_sizes = np.full(frames, length // frames, dtype=int)
        frame_sizes[: length % frames] += 1

        means = np.empty((frames, symbol_series.shape[1]))
        reconstruction = np.empty_like(symbol_series)

        start = 0
        for idx, size in enumerate(frame_sizes):
            end = start + size
            window = symbol_series[start:end]
            frame_mean = window.mean(axis=0)
            means[idx] = frame_mean
            reconstruction[start:end] = frame_mean
            start = end

        frame_values = means.mean(axis=1)
        symbol_indices = np.digitize(frame_values, self._breakpoints, right=False)
        symbol = tuple(int(i) for i in symbol_indices.tolist())

        error = float(np.mean((symbol_series - reconstruction) ** 2))
        return error, symbol

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default"):
        """Return parameter sets used by the aeon test suite."""
        if parameter_set == "default":
            return {
                "alphabet_size": 5,
                "paa_segments": 6,
                "min_segment_length": 15,
                "max_segment_length": 120,
                "num_lengths": 4,
                "penalty": 0.6,
            }
        raise ValueError(f"Unknown parameter_set '{parameter_set}'")
