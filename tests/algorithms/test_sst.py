"""Tests for the singular spectrum transformation detector (SSTDetector)."""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import SSTDetector
from tsseg.algorithms.sst.detector import _highest_peaks
from tsseg.algorithms.sst.sst import sst_scores


def _reference_scores(X: np.ndarray, N: int, lag: int, rank: int) -> np.ndarray:
    """Score of Ide & Tsuda (2007, Sec. 2), one explicit SVD per time point.

    Lagged vectors ``s(j) = (x_{j-N+1}, ..., x_j)``; past matrix ``[s(t - N),
    ..., s(t - 1)]`` (channels side by side), future matrix lagged by
    ``lag``; ``z(t) = 1 - ||U_r^T mu||^2``, 0 where the matrices do not fit.
    """
    X = np.asarray(X, dtype=float).reshape(len(X), -1)
    X = (X - X.mean(axis=0)) / X.std(axis=0) + 3.0
    T = X.shape[0]

    def hankel(t: int) -> np.ndarray:
        return np.hstack(
            [
                np.column_stack([X[j - N + 1 : j + 1, c] for j in range(t - N, t)])
                for c in range(X.shape[1])
            ]
        )

    z = np.zeros(T)
    for t in range(2 * N - 1, T - lag + 1):
        U = np.linalg.svd(hankel(t))[0][:, :rank]
        mu = np.linalg.svd(hankel(t + lag))[0][:, 0]
        z[t] = 1.0 - np.sum((U.T @ mu) ** 2)
    return z


def _frequency_change(n: int = 600, at: int = 300, seed: int = 0) -> np.ndarray:
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    x = np.where(t < at, np.sin(2 * np.pi * t / 40), np.sin(2 * np.pi * t / 15))
    return x + 0.1 * rng.standard_normal(n)


@pytest.mark.parametrize(("N", "lag", "rank"), [(20, 10, 3), (21, 5, 1), (15, 7, 5)])
def test_exact_score_matches_explicit_svd(N, lag, rank):
    x = _frequency_change(400)
    expected = _reference_scores(x, N, lag, rank)
    got = sst_scores(x[:, None], N, lag, rank, method="exact")
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_exact_score_multivariate_concatenates_channels():
    rng = np.random.default_rng(1)
    X = np.column_stack([_frequency_change(300), rng.standard_normal(300)])
    expected = _reference_scores(X, 15, 7, 3)
    got = sst_scores(X, 15, 7, 3, method="exact")
    np.testing.assert_allclose(got, expected, atol=1e-10)


@pytest.mark.parametrize("channels", [1, 3])
def test_fft_rsvd_score_close_to_exact(channels):
    rng = np.random.default_rng(2)
    X = np.column_stack(
        [_frequency_change(500, seed=c) for c in range(channels)]
    ) + 0.1 * rng.standard_normal((500, channels))
    exact = sst_scores(X, 30, 15, 5, method="exact")
    fast = sst_scores(X, 30, 15, 5, method="fft_rsvd", random_state=0)
    assert np.abs(fast - exact).max() < 1e-4
    assert np.all(fast[: 2 * 30 - 1] == 0) and np.all(fast[500 - 15 + 1 :] == 0)


def test_score_bounds_and_edges():
    x = _frequency_change(300)
    z = SSTDetector(n_cps=1, window_size=20, random_state=0)
    z.fit_predict(x)
    s = z.scores_
    assert s.shape == (300,)
    assert np.all((s >= 0) & (s <= 1))
    assert np.all(s[:39] == 0) and np.all(s[300 - 10 + 1 :] == 0)
    assert np.any(s[39 : 300 - 10 + 1] > 0)


@pytest.mark.parametrize("method", ["fft_rsvd", "exact"])
def test_finds_frequency_and_level_changes(method):
    rng = np.random.default_rng(3)
    t = np.arange(1500)
    x = np.where(t < 500, np.sin(2 * np.pi * t / 40), np.sin(2 * np.pi * t / 12))
    x = x + np.where(t < 1000, 0.0, 3.0) + 0.1 * rng.standard_normal(1500)
    cps = SSTDetector(n_cps=2, method=method, random_state=0).fit_predict(x)
    assert cps.tolist() == sorted(cps.tolist()) and len(cps) == 2
    assert 500 <= cps[0] <= 550 and 1000 <= cps[1] <= 1050


def test_univariate_1d_and_2d_agree():
    x = _frequency_change()
    a = SSTDetector(n_cps=2, window_size=20, method="exact").fit_predict(x)
    b = SSTDetector(n_cps=2, window_size=20, method="exact").fit_predict(x[:, None])
    np.testing.assert_array_equal(a, b)


def test_default_lag_is_half_window():
    x = _frequency_change()
    a = SSTDetector(n_cps=1, window_size=21, method="exact")
    b = SSTDetector(n_cps=1, window_size=21, lag=10, method="exact")
    np.testing.assert_array_equal(a.fit_predict(x), b.fit_predict(x))
    np.testing.assert_array_equal(a.scores_, b.scores_)


def test_random_state_reproducible_and_tag():
    x = _frequency_change()
    a = SSTDetector(n_cps=2, random_state=5).fit_predict(x)
    b = SSTDetector(n_cps=2, random_state=5).fit_predict(x)
    np.testing.assert_array_equal(a, b)
    assert SSTDetector(n_cps=1).get_tag("non_deterministic") is True
    assert SSTDetector(n_cps=1, random_state=0).get_tag("non_deterministic") is False
    assert SSTDetector(n_cps=1, method="exact").get_tag("non_deterministic") is False


def test_min_distance_between_change_points():
    x = _frequency_change(800, at=400)
    cps = SSTDetector(n_cps=3, window_size=20, min_distance=150, random_state=0)
    out = cps.fit_predict(x)
    assert len(out) <= 3
    assert np.all(np.diff(out) >= 150)
    assert np.all((out > 0) & (out < 800))


def test_n_cps_zero_and_required():
    x = _frequency_change()
    assert SSTDetector(n_cps=0, random_state=0).fit_predict(x).size == 0
    with pytest.raises(ValueError, match="n_cps"):
        SSTDetector(random_state=0).fit_predict(x)


def test_invalid_settings_rejected():
    x = _frequency_change(100)
    with pytest.raises(ValueError):
        SSTDetector(n_cps=1, window_size=50).fit_predict(x)  # needs 124 points
    with pytest.raises(ValueError):
        SSTDetector(n_cps=1, window_size=10, rank=10).fit_predict(x)
    with pytest.raises(ValueError):
        SSTDetector(n_cps=1, method="svd").fit_predict(x)


def test_constant_series_gives_no_error():
    cps = SSTDetector(n_cps=2, window_size=10, method="exact").fit_predict(np.ones(200))
    assert cps.dtype == np.int64 and np.all((cps > 0) & (cps < 200))


def test_peaks_at_the_ends_of_the_valid_range_are_kept():
    # a change close to an end leaves the score monotone up to that end
    rising = np.linspace(0.0, 1.0, 50)
    assert _highest_peaks(rising, 1, 10).tolist() == [49]
    assert _highest_peaks(rising[::-1], 1, 10).tolist() == [0]
    curve = np.r_[rising, rising[::-1][1:], 0.5 * rising]
    assert _highest_peaks(curve, 2, 10).tolist() == [49, 148]
    assert _highest_peaks(curve, 5, 10).tolist() == [49, 148]
