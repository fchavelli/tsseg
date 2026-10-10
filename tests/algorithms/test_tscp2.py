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
