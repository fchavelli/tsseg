"""Behavioural tests of ProphetDetector."""

import numpy as np
import pytest

pytest.importorskip("prophet")

from tsseg.algorithms import ProphetDetector  # noqa: E402

N = 1000
TRUE_CPS = np.array([250, 500, 750])


def _steps(rng=0, n_channels=1):
    rng = np.random.default_rng(rng)
    means = [0.0, 3.0, -1.0, 2.0]
    x = np.concatenate([rng.normal(m, 0.5, (250, n_channels)) for m in means])
    return x[:, 0] if n_channels == 1 else x


def _close(pred, truth, tol):
    return all(np.min(np.abs(np.asarray(pred) - t)) <= tol for t in truth)


def test_guided_finds_level_shifts():
    pred = ProphetDetector(n_changepoints=3).fit_predict(_steps())
    assert len(pred) == 3
    assert _close(pred, TRUE_CPS, 0.03 * N)


def test_output_depends_on_the_signal():
    """The old wrapper returned the same evenly spaced grid for any input."""
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(size=N))
    a = ProphetDetector(n_changepoints=3).fit_predict(_steps())
    b = ProphetDetector(n_changepoints=3).fit_predict(walk)
    grid = np.linspace(0, N - 1, 4).round()[1:]
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, grid)


def test_zero_change_points():
    assert ProphetDetector(n_changepoints=0).fit_predict(_steps()).size == 0


def test_minimum_distance():
    pred = ProphetDetector(n_changepoints=6, tolerance=0.05).fit_predict(_steps())
    assert np.all(np.diff(pred) >= 0.05 * N)


def test_unsupervised_threshold():
    det = ProphetDetector()
    pred = det.fit_predict(_steps())
    assert 0 < len(pred) < det.n_candidates
    assert np.all((pred > 0) & (pred < N))
    # Every returned candidate exceeds the threshold.
    strength = dict(zip(det.candidates_, det.changepoint_strengths_))
    assert all(strength[p] > det.delta_threshold for p in pred)


@pytest.mark.parametrize("strategy", ["ensembling", "l2"])
def test_multivariate_change_in_one_channel(strategy):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.5, (N, 3))
    x[600:, 1] += 3.0
    pred = ProphetDetector(
        n_changepoints=1, multivariate_strategy=strategy
    ).fit_predict(x)
    assert _close(pred, [600], 0.03 * N)


def test_block_reduction_keeps_full_series_indices():
    x = np.repeat(_steps(), 4)  # 4000 samples, change points at 1000, 2000, 3000
    pred = ProphetDetector(n_changepoints=3, max_points=1000).fit_predict(x)
    assert _close(pred, 4 * TRUE_CPS, 0.03 * len(x))
