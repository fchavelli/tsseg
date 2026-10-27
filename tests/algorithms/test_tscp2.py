"""Behavioural tests of TSCP2Detector (TensorFlow and ``tcn`` required)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("tcn")

from tsseg.algorithms import TSCP2Detector  # noqa: E402

# Small network and two epochs: each fit takes about a second on a CPU.
FAST = {"window_size": 16, "epochs": 2, "batch_size": 16}


def _mean_shift(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.0, 1.0, 150), rng.normal(3.0, 1.0, 150)])


@pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
def test_random_state_makes_training_reproducible(dropout_rate):
    """Two detectors with the same seed learn the same encoder."""
    X = _mean_shift()
    params = {"random_state": 0, "dropout_rate": dropout_rate, **FAST}
    first = TSCP2Detector(**params).fit(X)
    second = TSCP2Detector(**params).fit(X)
    # The automatic threshold is derived from the last training batch: it
    # differs between unseeded runs (weights and shuffling are random).
    assert first.similarity_threshold == second.similarity_threshold
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_random_state_is_reapplied_on_each_fit():
    """Refitting the same instance reproduces the first fit."""
    X = _mean_shift()
    detector = TSCP2Detector(random_state=3, **FAST)
    first = detector.fit_predict(X)
    threshold = detector.similarity_threshold
    second = detector.fit_predict(X)
    assert detector.similarity_threshold == threshold
    np.testing.assert_array_equal(first, second)


def test_random_state_is_a_parameter():
    detector = TSCP2Detector(random_state=7)
    assert detector.get_params()["random_state"] == 7
    assert TSCP2Detector().get_params()["random_state"] is None


def test_negative_random_state_is_rejected():
    with pytest.raises(ValueError, match="random_state"):
        TSCP2Detector(random_state=-1, **FAST).fit(_mean_shift())


# A series of exactly ``2 * window_size`` points holds a single window pair:
# ``fit`` halves the window it uses (it needs two pairs to contrast).
TWO_WINDOWS = 2 * FAST["window_size"]


def test_two_window_series_warns_instead_of_printing(capsys):
    detector = TSCP2Detector(random_state=0, **FAST)
    with pytest.warns(UserWarning, match="window_size"):
        detector.fit(_mean_shift()[:TWO_WINDOWS])
    assert "Warning" not in capsys.readouterr().out


def test_two_window_series_leaves_window_size_unchanged():
    detector = TSCP2Detector(random_state=0, **FAST)
    with pytest.warns(UserWarning, match="window_size"):
        detector.fit(_mean_shift()[:TWO_WINDOWS])
    assert detector.get_params()["window_size"] == FAST["window_size"]


def test_refit_after_a_short_series_uses_the_full_window():
    """A first fit on a short series does not shrink the window of the next."""
    X = _mean_shift()
    params = {"random_state": 0, "n_cps": 2, **FAST}
    detector = TSCP2Detector(**params)
    with pytest.warns(UserWarning, match="window_size"):
        detector.fit(X[:TWO_WINDOWS])
    expected = TSCP2Detector(**params).fit_predict(X)
    np.testing.assert_array_equal(detector.fit_predict(X), expected)


def test_predict_accepts_a_series_of_two_windows():
    """The encoder of ``fit`` scores the single pair of such a series."""
    detector = TSCP2Detector(random_state=0, n_cps=1, **FAST).fit(_mean_shift())
    cps = detector.predict(_mean_shift(1)[:TWO_WINDOWS])
    assert cps.tolist() == [FAST["window_size"]]


def test_predict_on_a_too_short_series_raises_and_keeps_the_detector():
    X = _mean_shift()
    detector = TSCP2Detector(random_state=0, n_cps=2, **FAST).fit(X)
    expected = detector.predict(X)
    with pytest.raises(ValueError, match=f"at least {TWO_WINDOWS} points"):
        detector.predict(X[: TWO_WINDOWS - 1])
    assert detector.get_params()["window_size"] == FAST["window_size"]
    np.testing.assert_array_equal(detector.predict(X), expected)


def test_refit_on_predict_halves_the_window_for_a_short_series():
    detector = TSCP2Detector(
        random_state=0, n_cps=1, refit_on_predict=True, **FAST
    ).fit(_mean_shift())
    with pytest.warns(UserWarning, match="window_size"):
        cps = detector.predict(_mean_shift(1)[:TWO_WINDOWS])
    assert cps.shape == (1,)
    assert detector.get_params()["window_size"] == FAST["window_size"]


# Selection of the change points on a known change score. Each change point
# raises the score of every pair centred less than ``window_size`` from it:
# below, two bumps of that half-width, at 151 and at 231 (a lower one).
CENTRES = FAST["window_size"] + 5 * np.arange((300 - TWO_WINDOWS) // 5 + 1)


def _two_bumps() -> np.ndarray:
    """Change score of each pair centre of a 300-point series (stride 5)."""
    width = FAST["window_size"]
    first = np.clip(1.0 - np.abs(CENTRES - 151) / width, 0.0, None)
    second = 0.5 * np.clip(1.0 - np.abs(CENTRES - 231) / width, 0.0, None)
    return first + second


def _with_score(detector: TSCP2Detector, score: np.ndarray) -> TSCP2Detector:
    """Make ``detector`` see ``score`` as the change score of its pairs."""
    import tensorflow as tf

    detector._similarity_diag_fn = lambda *_: tf.constant(1.0 - score)
    return detector


def test_guided_change_points_are_a_window_apart():
    """The neighbours of the highest centre do not take the second place."""
    detector = TSCP2Detector(random_state=0, n_cps=2, **FAST).fit(_mean_shift())
    cps = _with_score(detector, _two_bumps()).predict(_mean_shift())
    assert cps.tolist() == [151, 231]


def test_unsupervised_change_points_are_a_window_apart():
    """One change point per bump of centres below the threshold."""
    detector = TSCP2Detector(random_state=0, similarity_threshold=0.9, **FAST).fit(
        _mean_shift()
    )
    cps = _with_score(detector, _two_bumps()).predict(_mean_shift())
    assert cps.tolist() == [151, 231]


def test_guided_mode_returns_n_cps_change_points_when_too_few_are_spaced():
    """Closer centres complete the set when the series has too little room."""
    n_cps = 25  # 54 centres, 5 points apart: at most 14 are 16 points apart
    detector = TSCP2Detector(random_state=0, n_cps=n_cps, **FAST).fit(_mean_shift())
    cps = _with_score(detector, _two_bumps()).predict(_mean_shift())
    assert cps.size == n_cps
    assert {151, 231} <= set(cps.tolist())
