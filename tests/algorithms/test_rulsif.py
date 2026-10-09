"""Tests for the RuLSIF change-point detector (Liu et al., 2013)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from tsseg.algorithms import RuLSIFDetector
from tsseg.algorithms.rulsif.rulsif import (
    _squared_distances,
    cv_folds,
    relative_pe_divergence,
    rulsif_scores,
    split_kernels,
    subsequences,
)


def _kernels(U: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    G = _squared_distances(U)
    return np.exp(-G[None, :, :] / (2.0 * np.asarray(sigmas)[:, None, None] ** 2))


def _mean_shift(seed: int, n: int = 200, shift: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 1, n), rng.normal(shift, 1, n)])


# ----------------------------------------------------------------------
# Core estimator
# ----------------------------------------------------------------------


def test_subsequences_concatenate_channels():
    X = np.arange(12, dtype=float).reshape(6, 2)
    S = subsequences(X, 3)
    assert S.shape == (4, 6)
    np.testing.assert_array_equal(np.sort(S[1]), np.sort(X[1:4].ravel()))
    assert subsequences(np.arange(5.0), 2).shape == (4, 2)


def test_divergence_matches_closed_form():
    """Single candidate: theta = (H + lambda I)^-1 h and the PE_alpha formula."""
    rng = np.random.default_rng(0)
    U = rng.normal(size=(60, 4))
    K = _kernels(U, [1.5])
    alpha, lam = 0.2, 0.05
    est, s, q = relative_pe_divergence(
        K[:, :30, :30], K[:, 30:, :30], alpha, np.array([lam]), 5
    )
    A, B = K[0, :30, :30], K[0, 30:, :30]
    H = alpha * A.T @ A / 30 + (1 - alpha) * B.T @ B / 30
    theta = np.linalg.solve(H + lam * np.eye(30), A.mean(axis=0))
    gA, gB = A @ theta, B @ theta
    expected = (
        -alpha / 2 * np.mean(gA**2) - (1 - alpha) / 2 * np.mean(gB**2) + gA.mean() - 0.5
    )
    assert (s, q) == (0, 0)
    assert est == pytest.approx(expected, rel=1e-10)


def test_divergence_vanishes_for_identical_samples():
    """Same samples on both sides: g fits 1 exactly and PE_alpha is 0."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 3))
    K = _kernels(np.vstack([X, X]), [0.5])
    est, _, _ = relative_pe_divergence(
        K[:, :25, :25], K[:, 25:, :25], 0.1, np.array([1e-10]), 5
    )
    assert est == pytest.approx(0.0, abs=1e-6)


def test_divergence_approaches_true_value():
    """N(0, 1) against N(0, 2^2): the estimate is close to the integral."""
    alpha = 0.1
    y = np.linspace(-20, 20, 40001)
    p, q = norm.pdf(y, 0, 1), norm.pdf(y, 0, 2)
    mix = alpha * p + (1 - alpha) * q
    true = 0.5 * np.sum(mix * (p / mix - 1) ** 2) * (y[1] - y[0])
    rng = np.random.default_rng(2)
    n = 300
    U = np.vstack([rng.normal(0, 1, (n, 1)), rng.normal(0, 2, (n, 1))])
    d_med = np.median(np.sqrt(_squared_distances(U)[np.triu_indices(2 * n, 1)]))
    K = _kernels(U, [d_med])
    est, _, _ = relative_pe_divergence(
        K[:, :n, :n], K[:, n:, :n], alpha, np.array([0.1]), 5
    )
    assert true == pytest.approx(0.209, abs=1e-3)
    assert est == pytest.approx(true, abs=0.08)


def test_cross_validation_selects_from_the_grid():
    rng = np.random.default_rng(3)
    U = np.vstack([rng.normal(0, 1, (40, 2)), rng.normal(1, 1, (40, 2))])
    K = _kernels(U, [0.5, 1.0, 2.0])
    lambdas = np.array([1e-3, 1e-1, 10.0])
    est, s, q = relative_pe_divergence(K[:, :40, :40], K[:, 40:, :40], 0.1, lambdas, 4)
    assert 0 <= s < 3 and 0 <= q < 3
    refit, _, _ = relative_pe_divergence(
        K[[s], :40, :40], K[[s], 40:, :40], 0.1, lambdas[[q]], 4
    )
    assert est == pytest.approx(refit, rel=1e-12)


def test_cv_folds_partition_every_candidate():
    folds = cv_folds(50, 5, 25)
    assert folds.shape == (2, 25, 50)
    for side in range(2):
        for c in range(25):
            # ten samples per fold, as floor(p * 5 / 50) in RelULSIF.m
            assert np.array_equal(np.bincount(folds[side, c], minlength=5), [10] * 5)
    # a fresh permutation per candidate and per set, the same at every call
    assert not np.array_equal(folds[0, 0], folds[0, 1])
    assert not np.array_equal(folds[0, 0], folds[1, 0])
    assert np.array_equal(folds, cv_folds(50, 5, 25))


# Output of a line-by-line NumPy transcription of the authors' MATLAB code
# (change_detection.m, RelULSIF.m, lib/, github.com/anewgithubname/
# change_detection at 496042e): RelULSIF(YRef, YTest) on the standardised
# subsequences starting at t, with the folds of ``cv_folds`` in place of
# MATLAB's generator. The test set is the numerator, as in the MATLAB code.
_OFFICIAL = [
    (0.1, 1, 60, -0.03270825258975829),
    (0.1, 1, 140, 3.4431266076302647),
    (0.1, 3, 60, 0.3600968683127679),
    (0.1, 3, 140, 3.943260976986153),
    (0.5, 1, 60, -0.02561540186296407),
    (0.5, 1, 140, 0.3778581052847816),
    (0.5, 3, 60, 0.13178374330092102),
    (0.5, 3, 140, 0.41097049606383096),
]


@pytest.mark.parametrize("alpha, n_channels, t, expected", _OFFICIAL)
def test_divergence_matches_the_official_code(alpha, n_channels, t, expected):
    rng = np.random.default_rng(n_channels)
    scale = np.array([1.0, 5.0, 0.2][:n_channels])
    x = rng.standard_normal((400, n_channels)) * scale
    x[200:] += 1.0
    n = 50
    K = split_kernels(subsequences(x, 10)[t : t + 2 * n], [0.6, 0.8, 1.0, 1.2, 1.4])
    est, _, _ = relative_pe_divergence(
        K[:, n:, n:],
        K[:, :n, n:],
        alpha,
        10.0 ** np.arange(-3, 2),
        5,
        cv_folds(n, 5, 25),
    )
    assert est == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_split_kernels_standardise_and_use_the_median_scale():
    rng = np.random.default_rng(4)
    U = rng.normal(size=(20, 3)) * [1.0, 10.0, 0.1]
    K = split_kernels(U, [1.0])
    Z = U / U.std(axis=0, ddof=1)
    d2 = _squared_distances(Z)[np.triu_indices(20, 1)]
    sigma = np.sqrt(0.5 * np.median(d2))
    assert np.allclose(K[0], np.exp(-_squared_distances(Z) / (2 * sigma**2)))
    assert split_kernels(np.ones((10, 2)), [1.0]) is None


def test_scores_ignore_the_scale_of_each_channel():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(260, 2))
    X[130:, 0] += 2.0
    s = rulsif_scores(X, n=20, k=5)
    s_scaled = rulsif_scores(X * [1e3, 1e-2], n=20, k=5)
    assert np.allclose(s, s_scaled, equal_nan=True, rtol=1e-7, atol=1e-9)


def test_scores_are_aligned_and_undefined_at_the_ends():
    X = _mean_shift(0)
    k, n = 10, 50
    scores = rulsif_scores(X, k=k, n=n)
    valid = np.flatnonzero(~np.isnan(scores))
    offset = n + (k - 1) // 2
    assert valid[0] == offset
    assert valid[-1] == len(X) - 2 * n - k + 1 + offset
    assert np.all(np.isfinite(scores[valid]))
    assert abs(int(np.nanargmax(scores)) - 200) <= 10


def test_scores_too_short_series_are_all_nan():
    assert np.all(np.isnan(rulsif_scores(np.zeros(50), k=10, n=25)))


def test_constant_series_scores_zero():
    scores = rulsif_scores(np.ones(150), k=5, n=20)
    assert np.nanmax(np.abs(scores)) == 0.0


# ----------------------------------------------------------------------
# Detector
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_guided_mean_shift(seed):
    detector = RuLSIFDetector(n_cps=1)
    cps = detector.fit_predict(_mean_shift(seed))
    assert cps.dtype == np.int64
    assert len(cps) == 1 and abs(int(cps[0]) - 200) <= 10
    assert detector.scores_.shape == (400,)


def test_unsupervised_mean_shift_and_noise():
    cps = RuLSIFDetector().fit_predict(_mean_shift(0))
    assert len(cps) >= 1
    assert np.any(np.abs(cps - 200) <= 10)
    assert np.all(np.abs(cps - 200) <= 30)
    noise = np.random.default_rng(5).normal(size=400)
    # the calibrated threshold trades a few false alarms on noise (about 3
    # per 2,000 points) for recall; 3.0 is about the 99th percentile of the
    # score on change-free series
    assert len(RuLSIFDetector().fit_predict(noise)) <= 1
    assert len(RuLSIFDetector(threshold=3.0).fit_predict(noise)) == 0


def test_multivariate_change_in_one_channel():
    rng = np.random.default_rng(4)
    X = np.column_stack([rng.normal(size=400), _mean_shift(4)])
    cps = RuLSIFDetector(n_cps=1).fit_predict(X)
    assert len(cps) == 1 and abs(int(cps[0]) - 200) <= 10


def test_guided_returns_peaks_min_distance_apart():
    X = np.concatenate([_mean_shift(6), _mean_shift(7) + 8.0])
    detector = RuLSIFDetector(n_cps=3, min_distance=60)
    cps = detector.fit_predict(X)
    assert len(cps) == 3
    assert np.all(np.diff(cps) >= 60)
    assert any(abs(int(c) - 200) <= 10 for c in cps)
    assert any(abs(int(c) - 400) <= 10 for c in cps)
    assert any(abs(int(c) - 600) <= 10 for c in cps)


def test_threshold_above_the_bound_of_the_score_warns():
    # alpha = 0.5 bounds the score by 1, below the default threshold
    with pytest.warns(UserWarning, match="no change point"):
        cps = RuLSIFDetector(alpha=0.5, n_subsequences=20).fit_predict(
            _mean_shift(0, n=100)
        )
    assert cps.size == 0
    cps = RuLSIFDetector(alpha=0.5, n_subsequences=20, threshold=0.25).fit_predict(
        _mean_shift(0, n=100)
    )
    assert np.any(np.abs(cps - 100) <= 5)


def test_zero_change_points():
    assert RuLSIFDetector(n_cps=0).fit_predict(_mean_shift(0)).size == 0


def test_series_too_short_is_rejected():
    with pytest.raises(ValueError, match="subsequence"):
        RuLSIFDetector().fit_predict(np.zeros(100))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma_factors": ()},
        {"sigma_factors": (1.0, -1.0)},
        {"lambdas": (-0.1,)},
    ],
)
def test_invalid_grids_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RuLSIFDetector(**kwargs).fit_predict(np.zeros(200))


def test_too_many_folds_are_rejected():
    with pytest.raises(ValueError, match="fold"):
        RuLSIFDetector(n_subsequences=4, n_folds=5).fit_predict(np.zeros(200))
