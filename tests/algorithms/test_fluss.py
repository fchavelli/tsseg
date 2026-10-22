"""FLUSSDetector: CAC threshold (no known number of segments) and output order."""

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


@pytest.mark.parametrize("n_segments", [4, 5])
def test_known_number_of_segments_returns_sorted_change_points(n_segments):
    # stumpy.fluss returns the regime locations in extraction order (lowest
    # corrected arc curve first), e.g. [981, 1982, 880] for 4 segments here.
    det = FLUSSDetector(window_size=20, n_segments=n_segments)
    pred = det.fit_predict(_regimes())
    assert len(pred) == n_segments - 1
    assert np.all(np.diff(pred) > 0)
    np.testing.assert_array_equal(det.scores, det.profile[pred])


def test_l2_strategy_returns_sorted_change_points():
    x = _regimes()
    X = np.column_stack([x, 0.5 * x])
    det = FLUSSDetector(window_size=20, n_segments=4, multivariate_strategy="l2")
    pred = det.fit_predict(X)
    assert len(pred) == 3
    assert np.all(np.diff(pred) > 0)
