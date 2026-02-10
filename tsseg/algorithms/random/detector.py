"""Random detector producing synthetic states and change points."""

from __future__ import annotations

import numpy as np

from ..base import BaseSegmenter


class RandomDetector(BaseSegmenter):
    """Random detector emitting change points and state labels.

    The detector is primarily intended for testing pipelines. It can operate in a
    purely unsupervised fashion, where the number of change points and states is
    drawn at random (with bounds derived from available data), or in a semi-supervised
    mode where those quantities are given but locations and assignments remain random.
    """

    _tags = {
        "capability:missing_values": True,
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "semi_supervised": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        *,
        semi_supervised: bool = False,
        n_change_points: int | None = None,
        n_states: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if semi_supervised and n_change_points is None:
            raise ValueError("n_change_points must be provided when semi_supervised=True.")
        if semi_supervised and n_states is None:
            raise ValueError("n_states must be provided when semi_supervised=True.")

        if n_change_points is not None:
            n_change_points = int(n_change_points)
            if n_change_points < 0:
                raise ValueError("n_change_points must be non-negative.")

        if n_states is not None:
            n_states = int(n_states)
            if n_states <= 0:
                raise ValueError("n_states must be positive.")

        self.semi_supervised = semi_supervised
        self.n_change_points = n_change_points
        self.n_states = n_states
        self.random_state = random_state

        base_segments = (self.n_change_points + 1) if self.n_change_points is not None else 2
        base_segments = max(base_segments, 1)
        self.n_segments = max(base_segments, 1)

        self.breakpoints_: np.ndarray = np.array([], dtype=int)
        self.segment_states_: np.ndarray = np.array([], dtype=int)
        self.state_sequence_: np.ndarray = np.array([], dtype=int)
        self.n_states_drawn_: int | None = None
        self.n_change_points_drawn_: int | None = None
        self._n_timepoints_: int | None = None
        self._rng: np.random.Generator | None = None
        super().__init__(axis=0)

        if self.semi_supervised:
            self.set_tags(semi_supervised=True)

    def _fit(self, X, y=None):
        """Generate a random segmentation for the provided series."""
        self._rng = np.random.default_rng(self.random_state)
        self._n_timepoints_ = X.shape[0]
        self.breakpoints_, self.segment_states_, self.state_sequence_ = self._sample_segmentation(
            self._n_timepoints_
        )
        return self

    def _predict(self, X):
        """Return state labels for ``X`` as an np.ndarray."""
        n_timepoints = X.shape[0]

        if self._n_timepoints_ != n_timepoints:
            # Re-sample if prediction data differs from the fitted length.
            self.breakpoints_, self.segment_states_, self.state_sequence_ = self._sample_segmentation(
                n_timepoints
            )
            self._n_timepoints_ = n_timepoints

        return self.state_sequence_.copy()

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_segmentation(
        self, n_timepoints: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_timepoints <= 0:
            self.n_change_points_drawn_ = 0
            self.n_states_drawn_ = 0
            return (np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))

        rng = self._ensure_rng()

        max_possible_cps = max(0, n_timepoints - 1)
        n_cps = self._resolve_change_point_count(rng, max_possible_cps)

        if n_cps == 0:
            breakpoints = np.array([], dtype=int)
        else:
            if n_cps > max_possible_cps:
                raise ValueError(
                    f"Requested {n_cps} change points but only {max_possible_cps} are possible"
                )
            breakpoints = np.sort(rng.choice(np.arange(1, n_timepoints), size=n_cps, replace=False))

        n_segments = n_cps + 1
        n_states = self._resolve_state_count(rng, n_cps, n_segments)

        segment_states = self._sample_states(rng, n_segments, n_states)
        state_sequence = self._expand_states(segment_states, breakpoints, n_timepoints)

        self.n_change_points_drawn_ = n_cps
        self.n_states_drawn_ = n_states

        return breakpoints, segment_states, state_sequence

    def _resolve_change_point_count(
        self, rng: np.random.Generator, max_possible_cps: int
    ) -> int:
        if max_possible_cps == 0:
            return 0

        if self.semi_supervised:
            if self.n_change_points is None:
                raise ValueError("n_change_points must be set in semi-supervised mode.")
            if self.n_change_points > max_possible_cps:
                raise ValueError(
                    f"Requested {self.n_change_points} change points but at most {max_possible_cps} fit "
                    "the series length."
                )
            return self.n_change_points

        upper_bound = min(14, max_possible_cps)
        if upper_bound < 3:
            return upper_bound
        return int(rng.integers(3, upper_bound + 1))

    def _resolve_state_count(
        self,
        rng: np.random.Generator,
        n_cps: int,
        n_segments: int,
    ) -> int:
        if self.semi_supervised:
            if self.n_states is None:
                raise ValueError("n_states must be set in semi-supervised mode.")
            if n_segments > 1 and self.n_states < 2:
                raise ValueError("At least two states required when multiple segments exist.")
            return self.n_states

        lower = max(1, min(n_cps + 1, 10))
        upper = 10
        if lower > upper:
            lower = upper
        if lower == upper:
            return lower
        return int(rng.integers(lower, upper + 1))

    @staticmethod
    def _sample_states(
        rng: np.random.Generator, n_segments: int, n_states: int
    ) -> np.ndarray:
        if n_segments <= 0:
            return np.array([], dtype=int)
        if n_segments > 1 and n_states < 2:
            raise ValueError("At least two states are required for multiple segments.")

        states = np.zeros(n_segments, dtype=int)
        previous = None
        for idx in range(n_segments):
            if n_states == 1:
                states[idx] = 0
                previous = 0
                continue

            candidates = np.arange(n_states)
            if previous is not None:
                candidates = candidates[candidates != previous]
            states[idx] = int(rng.choice(candidates))
            previous = states[idx]

        return states

    @staticmethod
    def _expand_states(
        segment_states: np.ndarray, breakpoints: np.ndarray, n_timepoints: int
    ) -> np.ndarray:
        state_sequence = np.empty(n_timepoints, dtype=int)
        boundaries = np.concatenate(([0], breakpoints, [n_timepoints]))
        for start, end, state in zip(boundaries[:-1], boundaries[1:], segment_states):
            state_sequence[start:end] = state
        return state_sequence

    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.random_state)
        return self._rng
