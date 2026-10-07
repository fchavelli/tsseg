"""FLUSSDetector without a known number of segments (CAC threshold)."""

import numpy as np
import pytest

pytest.importorskip("stumpy")

from tsseg.algorithms import FLUSSDetector  # noqa: E402


def _regimes(rng=0):
    rng = np.random.default_rng(rng)
    t = np.arange(1000)
    x = np.concatenate([np.sin(t / 5), np.sign(np.sin(t / 7)), np.sin(t / 3)])
    return x + 0.1 * rng.normal(size=x.size)


def test_threshold_zero_finds_nothing():
    det = FLUSSDetector(window_size=20, n_segments=None, threshold=0.0)
    assert len(det.fit_predict(_regimes())) == 0


def test_threshold_finds_the_regime_changes():
    det = FLUSSDetector(window_size=20, n_segments=None, threshold=0.2)
    pred = det.fit_predict(_regimes())
    assert len(pred) >= 2
    for cp in (1000, 2000):
        assert np.min(np.abs(pred - cp)) <= 30
    assert np.all(det.profile[pred] < 0.2)
    assert np.all(np.diff(pred) >= 5 * 20)


def test_known_number_of_segments_is_unchanged():
    pred = FLUSSDetector(window_size=20, n_segments=3).fit_predict(_regimes())
    assert len(pred) == 2


def test_ensembling_without_k():
    x = _regimes()
    rng = np.random.default_rng(1)
    X = np.column_stack(
        [x, x + 0.05 * rng.normal(size=x.size), rng.normal(size=x.size)]
    )
    det = FLUSSDetector(window_size=20, n_segments=None, threshold=0.2, consensus=0.5)
    pred = det.fit_predict(X)
    for cp in (1000, 2000):
        assert np.min(np.abs(pred - cp)) <= 40
