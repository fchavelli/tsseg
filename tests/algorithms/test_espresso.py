"""EspressoDetector: ``random_state`` seeds the matrix profile computation."""

import numpy as np
import pytest

from tsseg.algorithms import EspressoDetector
from tsseg.algorithms.espresso.python.ESPRESSO_Script import espresso


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
