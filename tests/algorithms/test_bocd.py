"""BOCD against exhaustive Bayesian inference on short series."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from tsseg.algorithms.bocd import detector as bocd
from tsseg.algorithms.bocd._offline import (
    _boundary_marginals,
    _GaussianSegments,
    _map_partition,
)
from tsseg.algorithms.bocd.bayesian_models import _offline_changepoint_probabilities


def _segment_evidence(x, mu=0.0, kappa=1.0, alpha=1.0, beta=1.0):
    """Integrate the Gaussian mean and variance using a multivariate t density."""
    n = len(x)
    return multivariate_t.logpdf(
        x,
        loc=np.full(n, mu),
        shape=(beta / alpha) * (np.eye(n) + np.ones((n, n)) / kappa),
        df=2 * alpha,
    )


def _enumerate_boundaries(x, hazard, max_segment_length=None, **prior):
    """Sum all 2**(n-1) segmentations, with independent Bernoulli boundaries."""
    n = len(x)
    masks = np.array(list(product([False, True], repeat=n - 1)))
    weights = []
    for mask in masks:
        cuts = np.r_[0, np.flatnonzero(mask) + 1, n]
        if max_segment_length is not None and np.any(
            np.diff(cuts)[:-1] > max_segment_length
        ):
            weights.append(-np.inf)
            continue
        k = mask.sum()
        weight = k * np.log(hazard) + (n - 1 - k) * np.log1p(-hazard)
        weight += sum(
            _segment_evidence(x[start:end], **prior)
            for start, end in zip(cuts[:-1], cuts[1:], strict=True)
        )
        weights.append(weight)
    weights = np.asarray(weights)
    probabilities = np.exp(weights - logsumexp(weights))
    return probabilities @ masks


@pytest.mark.parametrize("n", [1, 2, 5])
def test_segment_evidence_matches_integrated_student_t(n):
    x = np.array([1.2, -0.3, 2.1, 0.2, -1.0])[:n]
    prior = {"mu": 0.7, "kappa": 1.3, "alpha": 2.2, "beta": 0.4}
    likelihood = bocd._StudentTLogMarginal(0.7, 1.3, 2.2, 0.4)
    assert likelihood.pdf(x, 0, n) == pytest.approx(_segment_evidence(x, **prior))


@pytest.mark.parametrize("hazard_lambda", [2, 10, 300, 1_000_000_000])
@pytest.mark.parametrize(
    "prior",
    [{}, {"mu": 0.7, "kappa": 1.3, "alpha": 2.2, "beta": 0.4}],
)
def test_boundary_probabilities_match_all_segmentations(
    monkeypatch, hazard_lambda, prior
):
    x = np.array([-1.0, -0.8, -1.2, 1.1, 0.8, 1.2, 0.7])
    captured = {}

    def capture(probabilities, **kwargs):
        captured["probabilities"] = probabilities
        return np.empty(0, dtype=np.int64)

    monkeypatch.setattr(bocd, "_select_change_points", capture)
    params = {"beta": 1, **prior}
    bocd.BOCDDetector(
        hazard_lambda=hazard_lambda,
        min_distance=1,
        decoding="marginal",
        standardize=False,
        max_candidates=None,
        **params,
    ).fit_predict(x)
    expected = _enumerate_boundaries(x, 1 / hazard_lambda, **prior)
    np.testing.assert_allclose(
        captured["probabilities"], expected, rtol=1e-10, atol=1e-12
    )


def test_more_than_twenty_certain_boundaries():
    """A marginal threshold of .05 does not cap detections at twenty."""
    rng = np.random.default_rng(7)
    means = np.tile([-10.0, 10.0], 15)
    x = np.repeat(means, 6) + rng.normal(0, 0.1, 180)
    result = bocd.BOCDDetector(
        hazard_lambda=6,
        kappa=0.001,
        beta=0.01,
        min_distance=3,
        decoding="marginal",
        standardize=False,
        cp_prob_threshold=0.05,
        max_candidates=None,
    ).fit_predict(x)
    np.testing.assert_array_equal(result, np.arange(6, 180, 6))


def test_truncated_messages_use_the_same_segments():
    """At truncate=0, only the first two nonterminal ends survive per start."""
    x = np.array([-1.0, -0.8, -1.2, 1.1, 0.8, 1.2, 0.7])
    likelihood = bocd._StudentTLogMarginal(0, 1, 1, 1)
    result = _offline_changepoint_probabilities(x, 0.1, likelihood, truncate=0)
    expected = _enumerate_boundaries(x, 0.1, max_segment_length=2)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_large_offset_keeps_boundary_probabilities():
    x = np.array([-1.0, -0.8, -1.2, 1.1, 0.8, 1.2, 0.7])
    offset = 2**30
    # Account for representational rounding in the shifted input itself.
    rounded = (x + offset) - offset
    expected = _enumerate_boundaries(rounded, 0.1)
    likelihood = bocd._StudentTLogMarginal(offset, 1, 1, 1)
    result = _offline_changepoint_probabilities(x + offset, 0.1, likelihood)
    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-12)


def test_uninformative_likelihood_recovers_constant_hazard(monkeypatch):
    """A geometric duration prior still has survival mass beyond lambda."""
    likelihood = bocd._StudentTLogMarginal(0, 1, 1, 1)
    monkeypatch.setattr(
        likelihood, "_log_marginals", lambda means, ss, counts: np.zeros(len(counts))
    )
    result = _offline_changepoint_probabilities(np.zeros(601), 1 / 300, likelihood)
    np.testing.assert_allclose(result, 1 / 300, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("n", [0, 1])
def test_no_interior_boundary(n):
    result = _offline_changepoint_probabilities(
        np.zeros(n), 0.1, bocd._StudentTLogMarginal(0, 1, 1, 1)
    )
    assert result.shape == (0,)


@pytest.mark.parametrize("strategy", ["l2", "ensembling"])
def test_zero_cap_skips_inference(monkeypatch, strategy):
    def fail(*args, **kwargs):
        pytest.fail("No inference is needed when max_cps=0")

    monkeypatch.setattr(bocd, "_GaussianSegments", fail)
    x = np.random.default_rng(7).normal(size=(80, 2))
    assert (
        bocd.BOCDDetector(max_cps=0, multivariate_strategy=strategy).fit_predict(x).size
        == 0
    )


def _niw_evidence(x, mu=0.0, kappa=1.0, alpha=1.0, beta=1.0):
    """Independent oracle: multiply sequential multivariate-t predictions."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    d = x.shape[1]
    mean = np.full(d, mu)
    nu = 2 * alpha + d - 1
    scale = 2 * beta * np.eye(d)
    evidence = 0.0
    for point in x:
        df = nu - d + 1
        evidence += multivariate_t.logpdf(
            point, loc=mean, shape=scale * (kappa + 1) / (kappa * df), df=df
        )
        deviation = point - mean
        scale += (kappa / (kappa + 1)) * np.outer(deviation, deviation)
        mean += deviation / (kappa + 1)
        kappa += 1
        nu += 1
    return evidence


def _niw_partitions(x, positions, min_size, hazard, cap=None, **prior):
    n = len(x)
    partitions, weights = [], []
    cache = {}
    for mask in product([False, True], repeat=len(positions) - 2):
        cp = positions[1:-1][np.asarray(mask, dtype=bool)]
        cuts = np.r_[0, cp, n]
        if np.any(np.diff(cuts) < min_size) or (cap is not None and len(cp) > cap):
            continue
        value = len(cp) * np.log(hazard) + (n - 1 - len(cp)) * np.log1p(-hazard)
        for start, end in zip(cuts[:-1], cuts[1:], strict=True):
            if (start, end) not in cache:
                cache[start, end] = _niw_evidence(x[start:end], **prior)
            value += cache[start, end]
        partitions.append(cp)
        weights.append(value)
    weights = np.asarray(weights)
    probabilities = np.zeros(n - 1)
    for cp, weight in zip(
        partitions, np.exp(weights - logsumexp(weights)), strict=True
    ):
        probabilities[cp - 1] += weight
    return partitions[int(np.argmax(weights))], probabilities


@pytest.mark.parametrize("d", [1, 2, 4])
@pytest.mark.parametrize("limit", [None, 2])
def test_native_evidence_matches_sequential_predictive(d, limit):
    x = np.random.default_rng(12).normal(size=(9, d))
    params = {"mu0": 0.7, "kappa0": 1.3, "alpha0": 2.2, "beta0": 0.4}
    segments = _GaussianSegments(x, max_candidates=limit, **params)
    for reverse in [False, True]:
        for endpoint, neighbours, evidence in segments.rows(1, reverse=reverse):
            for neighbour, actual in zip(neighbours, evidence, strict=True):
                start, end = (endpoint, neighbour) if reverse else (neighbour, endpoint)
                a, b = segments.positions[[start, end]]
                expected = _niw_evidence(x[a:b], mu=0.7, kappa=1.3, alpha=2.2, beta=0.4)
                assert actual == pytest.approx(expected, rel=1e-10, abs=1e-11)


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("minimum", [1, 2])
@pytest.mark.parametrize("limit", [None, 2])
def test_native_messages_and_count_constraint_match_all_partitions(d, minimum, limit):
    x = np.random.default_rng(2).normal(size=(8, d))
    x[4:] += 3
    segments = _GaussianSegments(
        x, mu0=0, kappa0=1, alpha0=1, beta0=1, max_candidates=limit
    )
    for cap in [None, 0, 1, 2]:
        expected, _ = _niw_partitions(x, segments.positions, minimum, 0.2, cap)
        actual = _map_partition(segments, 0.2, minimum, cap)
        np.testing.assert_array_equal(actual, expected)
    _, expected = _niw_partitions(x, segments.positions, minimum, 0.2)
    actual = _boundary_marginals(segments, 0.2, minimum, None)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)


def test_native_offset_and_rank_deficiency():
    x = np.c_[np.linspace(-1, 1, 40), np.zeros(40)]
    offset = 2**30
    rounded = (x + offset) - offset

    def run(values, mu):
        obj = _GaussianSegments(
            values, mu0=mu, kappa0=1, alpha0=1, beta0=1e-6, max_candidates=15
        )
        return _boundary_marginals(obj, 0.1, 3, None)

    np.testing.assert_allclose(run(x + offset, offset), run(rounded, 0), atol=1e-9)


def test_candidate_grid_keeps_all_observations():
    x = np.array([0.0, 0.0, 9.0, 0.0, 0.0, -7.0, 0.0, 0.0, 0.0])
    obj = _GaussianSegments(x, mu0=0, kappa0=1, alpha0=1, beta0=1, max_candidates=2)
    assert len(obj.positions) <= 4
    last = list(obj.rows(1))[-1]
    assert last[2][0] == pytest.approx(_niw_evidence(x), abs=1e-11)


def test_native_retains_tiny_prior_on_rank_one_scatter():
    beta = 1e-30
    obj = _GaussianSegments(
        np.zeros((8, 3)), mu0=0, kappa0=1, alpha0=1, beta0=beta, max_candidates=None
    )
    actual = obj._log_marginals(
        np.zeros((1, 3)), np.full((1, 3, 3), 4.0), np.array([8])
    )[0]
    # Scatter eigenvalues are exactly (0, 0, 12); the prior supplies both
    # zero directions, even below the scale at which matrix addition resolves it.
    logdet = 2 * np.log(2 * beta) + np.log(12 + 2 * beta)
    expected = obj._normalisers[8] - 0.5 * (obj.nu + 8) * logdet
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-10)


def test_native_detects_covariance_change_lost_by_l2():
    rng = np.random.default_rng(8)
    block = rng.normal(size=(300, 2)) @ np.linalg.cholesky([[1, 0.95], [0.95, 1]]).T
    x = np.r_[block, block * [1, -1]]
    native = bocd.BOCDDetector(max_cps=1).fit_predict(x)
    assert len(native) == 1 and abs(native[0] - 300) <= 10
    l2 = bocd.BOCDDetector(multivariate_strategy="l2", max_cps=1).fit_predict(x)
    assert l2.size == 0


def test_standardization_transfers_to_channel_units():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(240, 2))
    x[120:] += [5, -3]
    expected = bocd.BOCDDetector().fit_predict(x)
    actual = bocd.BOCDDetector().fit_predict(x * [100, 0.01] + [10000, -50])
    assert expected.size > 0
    np.testing.assert_array_equal(actual, expected)


def test_ensembling_spaces_the_final_union(monkeypatch):
    locations = iter([np.array([40, 100]), np.array([41, 101])])
    monkeypatch.setattr(bocd.BOCDDetector, "_run_bocd", lambda *args: next(locations))
    x = np.random.default_rng(0).normal(size=(160, 2))
    result = bocd.BOCDDetector(multivariate_strategy="ensembling").fit_predict(x)
    np.testing.assert_array_equal(result, [40, 100])
