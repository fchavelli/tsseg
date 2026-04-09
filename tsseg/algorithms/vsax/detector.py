"""Variable-length SAX baseline detector.

Segmentation via dynamic programming over per-channel SAX symbols with
agglomerative symbol clustering.  Reconstruction error is computed in O(1)
per candidate via prefix sums.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from ..base import BaseSegmenter
from .breakpoints import get_breakpoints


class VSAXDetector(BaseSegmenter):
    """Baseline for state detection using variable-length SAX symbols.

    The detector uses dynamic programming over variable-length Symbolic
    Aggregate approXimation (SAX) representations to find the segmentation
    that minimises PAA reconstruction error with an additive penalty
    controlling fragmentation.

    SAX symbols are computed **per channel**, preserving multivariate
    structure.  Similar symbols are merged into the same state via
    agglomerative clustering on Hamming distance, avoiding the brittleness
    of exact symbol matching.

    Parameters
    ----------
    axis : int, default=0
        Time axis.  ``axis=0`` assumes ``(n_timepoints, n_channels)`` input.
    alphabet_size : int, default=6
        Number of SAX symbols per channel.  Values >= 1 are supported.
    paa_segments : int, default=8
        Number of PAA frames per segment.  Short segments automatically
        reduce the number of frames so that every frame contains at least
        one sample; the resulting symbol is zero-padded (by repeating the
        last frame) to a fixed length of ``paa_segments * n_channels``.
    min_segment_length : int, default=20
        Minimum admissible segment length (in samples).
    max_segment_length : int, default=180
        Maximum admissible segment length.
    num_lengths : int, default=6
        Number of candidate lengths linearly spaced between ``min`` and
        ``max``.  Increasing this value improves flexibility at the cost of
        runtime.
    penalty : float, default=0.8
        Cost added for every new segment.  Use larger values to favour
        longer segments; reduce to obtain more change points.
    symbol_merge_threshold : float, default=0.2
        Normalised distance threshold below which two SAX symbols are merged
        into the same state.  Distance is measured as mean absolute difference
        of symbol indices divided by ``alphabet_size`` (so it lies in [0, 1]).
        ``0`` gives exact matching (original behaviour), ``1`` collapses
        everything into a single state.
    zscore : bool, default=True
        Apply per-channel z-normalisation before computing scores.
    adaptive_breakpoints : bool, default=True
        When ``True``, learn SAX breakpoints from empirical quantiles of
        the training data instead of using Gaussian breakpoints.
    random_state : int | None, default=None
        Accepted for API compatibility but unused (deterministic).
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "capability:unsupervised": True,
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
        symbol_merge_threshold: float = 0.2,
        zscore: bool = True,
        adaptive_breakpoints: bool = True,
        random_state: int | None = 0,
    ) -> None:
        if alphabet_size < 1:
            raise ValueError("alphabet_size must be an integer >= 1")
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
        if not 0 <= symbol_merge_threshold <= 1:
            raise ValueError("symbol_merge_threshold must be in [0, 1]")

        self.alphabet_size = alphabet_size
        self.paa_segments = paa_segments
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.num_lengths = num_lengths
        self.penalty = penalty
        self.symbol_merge_threshold = symbol_merge_threshold
        self.zscore = zscore
        self.adaptive_breakpoints = adaptive_breakpoints
        self.random_state = random_state

        super().__init__(axis=axis)

    # ------------------------------------------------------------------
    # tsseg estimator API
    # ------------------------------------------------------------------
    def _fit(self, X: np.ndarray, y: np.ndarray | None = None, axis: int | None = None):
        X = self._validate_data(X)
        self._n_channels = X.shape[1]

        if self.adaptive_breakpoints and X.shape[0] > 0:
            data = self._z_normalise(X) if self.zscore else X.astype(float, copy=False)
            quantiles = np.linspace(0, 100, self.alphabet_size + 1)[1:-1]
            # Per-channel breakpoints: shape (alphabet_size - 1, n_channels)
            bp = np.percentile(data, quantiles, axis=0)
            if bp.ndim == 1:
                bp = bp[:, np.newaxis]
            self._breakpoints = bp
        else:
            bp = get_breakpoints(self.alphabet_size)  # (alphabet_size - 1,)
            self._breakpoints = np.tile(bp[:, np.newaxis], (1, self._n_channels))

        return self

    def _predict(self, X: np.ndarray, axis: int | None = None) -> np.ndarray:
        X = self._validate_data(X)
        n, d = X.shape

        if n == 0:
            return np.array([], dtype=int)

        series = self._z_normalise(X) if self.zscore else X.astype(float, copy=True)

        # ----- Prefix sums for O(1) segment statistics -----
        cumsum = np.zeros((n + 1, d), dtype=np.float64)
        cumsum[1:] = np.cumsum(series, axis=0)
        cumsum2 = np.zeros((n + 1, d), dtype=np.float64)
        cumsum2[1:] = np.cumsum(series ** 2, axis=0)

        candidate_lengths = self._candidate_lengths(n)
        if len(candidate_lengths) == 0:
            return np.zeros(n, dtype=int)

        # Pre-compute relative frame boundaries for each candidate length
        # so that np.linspace is not called inside the inner loop.
        bounds_cache: dict[int, np.ndarray] = {}
        for length in candidate_lengths:
            frames = min(self.paa_segments, int(length))
            bounds_cache[int(length)] = (
                np.linspace(0, length, frames + 1).round().astype(np.intp)
            )

        symbol_len = self.paa_segments * d

        # ----- DP backward -----
        dp_cost = np.full(n + 1, np.inf, dtype=np.float64)
        dp_end = np.full(n + 1, -1, dtype=np.intp)
        dp_symbols = np.zeros((n + 1, symbol_len), dtype=np.int32)
        dp_cost[n] = 0.0

        for start in range(n - 1, -1, -1):
            best_cost = np.inf
            best_end = -1
            best_symbol: np.ndarray | None = None
            max_len = n - start

            for length in self._enumerate_lengths(candidate_lengths, max_len):
                end = start + length
                rel_bounds = bounds_cache.get(length)
                if rel_bounds is None:
                    frames = min(self.paa_segments, length)
                    rel_bounds = np.linspace(0, length, frames + 1).round().astype(np.intp)
                cost, symbol = self._segment_cost_fast(
                    cumsum, cumsum2, start, d, rel_bounds,
                )
                total = cost + self.penalty + dp_cost[end]
                if total < best_cost:
                    best_cost = total
                    best_end = end
                    best_symbol = symbol

            if best_end == -1:
                # Fallback: consume entire remaining suffix
                remaining = max_len
                frames = min(self.paa_segments, remaining)
                rel_bounds = np.linspace(0, remaining, frames + 1).round().astype(np.intp)
                cost, symbol = self._segment_cost_fast(
                    cumsum, cumsum2, start, d, rel_bounds,
                )
                best_cost = cost + self.penalty
                best_end = n
                best_symbol = symbol

            dp_cost[start] = best_cost
            dp_end[start] = best_end
            dp_symbols[start] = best_symbol  # type: ignore[assignment]

        # ----- Reconstruct segments -----
        segments: list[tuple[int, int, np.ndarray]] = []
        idx = 0
        while idx < n:
            end = int(dp_end[idx])
            if end <= idx:
                segments.append((idx, n, dp_symbols[idx]))
                break
            segments.append((idx, end, dp_symbols[idx]))
            idx = end

        return self._cluster_and_label(segments, n)

    # ------------------------------------------------------------------
    # Vectorised helpers
    # ------------------------------------------------------------------
    def _segment_cost_fast(
        self,
        cumsum: np.ndarray,
        cumsum2: np.ndarray,
        seg_start: int,
        n_channels: int,
        rel_bounds: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """PAA reconstruction MSE + per-channel SAX symbol via prefix sums.

        Cost is O(frames * channels) regardless of segment length.
        """
        bounds = rel_bounds + seg_start
        length = int(bounds[-1] - bounds[0])
        sym_len = self.paa_segments * n_channels

        if length == 0:
            return 0.0, np.zeros(sym_len, dtype=np.int32)

        a = bounds[:-1]
        b = bounds[1:]
        sizes = b - a
        valid = sizes > 0
        a, b, sizes = a[valid], b[valid], sizes[valid]
        actual_frames = len(a)

        if actual_frames == 0:
            return 0.0, np.zeros(sym_len, dtype=np.int32)

        sizes_2d = sizes[:, np.newaxis].astype(np.float64)

        frame_sums = cumsum[b] - cumsum[a]      # (actual_frames, d)
        frame_sums2 = cumsum2[b] - cumsum2[a]   # (actual_frames, d)
        frame_means = frame_sums / sizes_2d     # (actual_frames, d)
        frame_vars = frame_sums2 / sizes_2d - frame_means ** 2
        np.maximum(frame_vars, 0.0, out=frame_vars)

        total_error = float(np.sum(frame_vars * sizes_2d))
        mse = total_error / (length * n_channels)

        # Per-channel SAX symbol: count how many breakpoints each mean exceeds
        # breakpoints: (A-1, d)   frame_means: (actual_frames, d)
        if self._breakpoints.shape[0] > 0:
            sym = (
                frame_means[:, np.newaxis, :]
                >= self._breakpoints[np.newaxis, :, :]
            ).sum(axis=1).astype(np.int32)          # (actual_frames, d)
        else:
            sym = np.zeros((actual_frames, n_channels), dtype=np.int32)

        # Pad shorter segments to fixed symbol length (paa_segments * d)
        if actual_frames < self.paa_segments:
            pad = np.tile(sym[-1:], (self.paa_segments - actual_frames, 1))
            sym = np.vstack([sym, pad])

        return mse, sym.ravel()

    def _cluster_and_label(
        self,
        segments: list[tuple[int, int, np.ndarray]],
        n_samples: int,
    ) -> np.ndarray:
        """Assign state labels by clustering similar SAX symbols."""
        if not segments:
            return np.zeros(n_samples, dtype=int)

        # Collect unique symbols
        sym_map: dict[tuple, int] = {}
        seg_keys: list[tuple] = []
        for _start, _end, sym in segments:
            key = tuple(sym.tolist())
            if key not in sym_map:
                sym_map[key] = len(sym_map)
            seg_keys.append(key)

        unique_keys = list(sym_map.keys())
        n_unique = len(unique_keys)

        if n_unique <= 1:
            return np.zeros(n_samples, dtype=int)

        # Agglomerative clustering on normalised Manhattan distance
        # (accounts for the ordinal nature of SAX symbols: a ±1 shift is
        # much less significant than a ±3 shift)
        if self.symbol_merge_threshold > 0:
            sym_array = np.array(unique_keys, dtype=np.float64)
            dist = pdist(sym_array, metric="cityblock")
            # Normalise to [0, 1]: max possible per-element diff is
            # alphabet_size, and there are sym_len elements.
            max_dist = max(self.alphabet_size, 1) * sym_array.shape[1]
            if max_dist > 0:
                dist /= max_dist
            Z = linkage(dist, method="complete")
            clusters = fcluster(
                Z, t=self.symbol_merge_threshold, criterion="distance",
            )
            key_to_cluster = {
                k: int(clusters[i]) for i, k in enumerate(unique_keys)
            }
        else:
            # Exact matching (backward-compatible)
            key_to_cluster = {k: i for i, k in enumerate(unique_keys)}

        # Re-number states sequentially by order of first appearance
        state_remap: dict[int, int] = {}
        next_state = 0
        labels = np.empty(n_samples, dtype=int)
        for (start, end, _sym), key in zip(segments, seg_keys, strict=True):
            raw = key_to_cluster[key]
            if raw not in state_remap:
                state_remap[raw] = next_state
                next_state += 1
            labels[start:end] = state_remap[raw]

        return labels

    # ------------------------------------------------------------------
    # Shared helpers
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
        self, candidate_lengths: Iterable[int], max_length: int,
    ) -> Iterable[int]:
        yielded = False
        for length in candidate_lengths:
            if length <= max_length:
                yielded = True
                yield int(length)
        if not yielded:
            yield max_length

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
                "symbol_merge_threshold": 0.2,
            }
        raise ValueError(f"Unknown parameter_set '{parameter_set}'")
