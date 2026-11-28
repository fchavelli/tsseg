"""ESPRESSO matrix profiles contain Euclidean distances, not their roots."""

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from tsseg.algorithms.espresso.python.calculateSemanticDensityMatrix import (
    calculate_semantic_density_matrix,
)
from tsseg.algorithms.espresso.python.ESPRESSO_Script import compute_mp
from tsseg.algorithms.espresso.python.MatrixProfile.timeseriesSelfJoinFast import (
    timeseries_self_join_fast,
)


def _direct_profile(x, m):
    """Exhaustive pairwise distances, independently of the FFT implementation."""
    windows = sliding_window_view(x, m)
    normalized = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(
        axis=1, keepdims=True
    )
    distances = np.linalg.norm(normalized[:, None] - normalized[None, :], axis=2)
    starts = np.arange(len(windows))
    # Use windows whose exclusion radius also agrees with the legacy rounding.
    distances[np.abs(starts[:, None] - starts[None, :]) <= (m + 2) // 4] = np.inf
    return distances.min(axis=1), distances.argmin(axis=1)


@pytest.mark.parametrize("m", [4, 7, 16, 32, 64, 128])
@pytest.mark.parametrize("seed", [0, 7])
def test_matrix_profile_matches_direct_euclidean_distances(m, seed):
    x = np.random.default_rng(91).normal(size=2 * m + 17)
    expected, expected_indices = _direct_profile(x, m)
    actual, indices = timeseries_self_join_fast(x, m, rng=seed)

    # Continuous noise avoids ties: removing an outer root must not move links.
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_multichannel_profiles_keep_euclidean_units():
    x = np.random.default_rng(15).normal(size=(2, 83))
    profiles, indices = compute_mp(x, 16, rng=0)
    for channel in range(len(x)):
        expected, expected_indices = _direct_profile(x[channel], 16)
        np.testing.assert_array_equal(indices[channel], expected_indices)
        np.testing.assert_allclose(profiles[channel], expected, atol=1e-10)


@pytest.mark.parametrize("chain_len", [0, 1, 2, 3, 5])
def test_chain_budget_is_sensitive_to_distance_units_from_round_three(chain_len):
    # Abstract arc costs isolate the downstream consequence of a second root.
    # This is not claimed to be a profile realized by a particular time series.
    distances = np.full(500, 4.0)
    distances[-1] = 9.0
    indices = (np.arange(500) + 60) % 500
    euclidean, _ = calculate_semantic_density_matrix(distances, indices, chain_len, 32)
    rooted, _ = calculate_semantic_density_matrix(
        np.sqrt(distances), indices, chain_len, 32
    )
    # Two hops cost at most 2*max(profile) in either representation. In the
    # third expansion four-hop costs can be filtered: 4*4 <= 2*9, whereas
    # 4*sqrt(4) > 2*sqrt(9). Check the primary density consumed by ESPRESSO.
    if chain_len <= 2:
        np.testing.assert_array_equal(euclidean, rooted)
    else:
        assert np.max(np.abs(euclidean - rooted)) > 1.0
