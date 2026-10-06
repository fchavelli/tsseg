"""Behavioural tests of BeastDetector (need Rbeast >= 0.1.25)."""

import numpy as np
import pytest

pytest.importorskip("Rbeast")

from tsseg.algorithms import BeastDetector  # noqa: E402


def _step(n=1000, at=500, rng=0):
    rng = np.random.default_rng(rng)
    x = rng.normal(0, 0.3, n)
    x[at:] += 3.0
    return x


def test_change_point_is_first_index_of_new_segment():
    """Rbeast counts time from 1: the CP used to come out one sample late."""
    assert BeastDetector().fit_predict(_step()).tolist() == [500]


def test_change_point_with_deltat():
    assert BeastDetector(deltat=0.5).fit_predict(_step()).tolist() == [500]


def test_multivariate_native():
    x = _step()
    X = np.c_[x, x + np.random.default_rng(1).normal(0, 0.3, len(x))]
    assert BeastDetector().fit_predict(X).tolist() == [500]


def test_long_univariate_series_returns():
    """Rbeast 0.1.15 never returned beyond about 4,600 points."""
    cps = BeastDetector().fit_predict(_step(n=6000, at=3000))
    assert 3000 in cps


def test_default_seed_is_reproducible():
    x = np.random.default_rng(2).normal(size=800)
    x[400:] += 1.0
    a = BeastDetector().fit_predict(x)
    b = BeastDetector().fit_predict(x)
    np.testing.assert_array_equal(a, b)
