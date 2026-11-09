"""BOCD against exhaustive Bayesian inference on short series."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_t

from tsseg.algorithms.bocd import detector as bocd
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


@pytest.mark.parametrize("hazard_lambda", [2, 10, 300])
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
    bocd.BOCDDetector(hazard_lambda=hazard_lambda, min_distance=1, **prior).fit_predict(
        x
    )
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
        hazard_lambda=6, kappa=0.001, beta=0.01, min_distance=3
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

    monkeypatch.setattr(bocd, "_offline_changepoint_probabilities", fail)
    x = np.random.default_rng(7).normal(size=(80, 2))
    assert (
        bocd.BOCDDetector(max_cps=0, multivariate_strategy=strategy).fit_predict(x).size
        == 0
    )
