"""EspressoDetector: ``random_state`` seeds the matrix profile computation."""

import numpy as np
import pytest

from tsseg.algorithms import EspressoDetector
from tsseg.algorithms.espresso.python.calculateSemanticDensityMatrix import (
    _extract_new_arc_set,
)
from tsseg.algorithms.espresso.python.ESPRESSO_Script import espresso
from tsseg.algorithms.espresso.python.MatrixProfile.timeseriesSelfJoinFast import (
    timeseries_self_join_fast,
)


def _periodic():
    """Three regimes built from exactly repeated periods.

    Repeated subsequences are bit-identical, so many matrix profile distances
    tie; the nearest-neighbour index kept for a tie depends on the random order
    in which the subsequences are visited, and so does the output.
    """
    fast = np.tile(np.sin(2 * np.pi * np.arange(10) / 10), 15)
    slow = np.tile(np.sin(2 * np.pi * np.arange(25) / 25), 6)
    return np.concatenate([fast, slow, fast])


def _multivariate():
    x = _periodic()
    return np.stack([x, np.roll(x, 7)], axis=1)


def _core(X, seed):
    """Reference: the ESPRESSO routine called directly with ``random_state``."""
    data = X[np.newaxis, :] if X.ndim == 1 else X.T
    return espresso(3, data, 10, 3, random_state=seed)


def _detector(seed):
    return EspressoDetector(window_size=10, n_segments=3, random_state=seed)


def test_the_seed_matters_on_the_periodic_signal():
    outputs = {tuple(_core(_periodic(), seed)) for seed in range(4)}
    assert len(outputs) > 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_state_seeds_the_detector(seed):
    x = _periodic()
    np.testing.assert_array_equal(_detector(seed).fit_predict(x), _core(x, seed))


@pytest.mark.parametrize("make_X", [_periodic, _multivariate])
def test_same_random_state_gives_the_same_output(make_X):
    X = make_X()
    detector = _detector(0)
    first = detector.fit_predict(X)
    np.testing.assert_array_equal(detector.predict(X), first)
    for _ in range(3):
        np.testing.assert_array_equal(_detector(0).fit_predict(X), first)


@pytest.mark.parametrize("m", [4, 5, 6, 7, 10, 18, 32])
def test_profile_neighbours_match_direct_distances_with_matlab_exclusion(m):
    """Use a direct distance oracle, including MATLAB's half-away tie rule."""
    signal = np.sin(np.arange(100) * 0.1)
    signal += np.random.default_rng(3).normal(0, 0.01, signal.size)
    windows = np.lib.stride_tricks.sliding_window_view(signal, m)
    normalized = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(
        axis=1, keepdims=True
    )
    distances = ((normalized[:, None] - normalized[None, :]) ** 2).sum(axis=2)
    for i in range(len(windows)):
        for j in range(len(windows)):
            # MATLAB round(m/4) rounds positive half-integers upwards.
            if abs(i - j) <= np.floor(m / 4 + 0.5):
                distances[i, j] = np.inf
    _, indices = timeseries_self_join_fast(signal, m, rng=0)
    np.testing.assert_array_equal(indices, distances.argmin(axis=1))


@pytest.mark.parametrize("m", range(4, 33))
@pytest.mark.parametrize("direction", [-1, 1])
def test_chain_separation_is_compared_to_the_unrounded_quarter_window(m, direction):
    """A two-hop arc is admitted exactly when its separation exceeds m/4."""
    n = 100
    start, intermediate = 40, 90
    for gap in range(1, m + 1):
        candidate = start + direction * gap
        indices = np.full(n, n, dtype=int)
        indices[start] = intermediate
        indices[intermediate] = candidate
        costs = np.ones(n)
        _, _, next_arcs, _ = _extract_new_arc_set(
            costs,
            indices,
            [[] for _ in range(n)],
            [[] for _ in range(n)],
            10.0,
            indices,
            costs,
            m,
            n,
        )
        expected = candidate if gap > m / 4 else n
        assert next_arcs[start] == expected, (m, gap, direction)


def test_threshold_sensitive_window_uses_upstream_chain_rule_end_to_end():
    signal = np.sin(np.arange(120) * 0.1)
    signal += np.random.default_rng(0).normal(0, 0.01, signal.size)
    expected = np.array([33, 99])
    np.testing.assert_array_equal(
        espresso(3, signal[np.newaxis], 6, 3, random_state=0), expected
    )
    np.testing.assert_array_equal(
        EspressoDetector(
            window_size=6, chain_len=3, n_segments=3, random_state=0
        ).fit_predict(signal),
        expected,
    )
