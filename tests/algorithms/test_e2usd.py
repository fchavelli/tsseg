"""Behaviour of E2USDDetector not covered by the generic suite."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsseg.algorithms import E2USDDetector  # noqa: E402

SMALL = {
    "window_size": 32,
    "step": 16,
    "n_states": 2,
    "nb_steps": 2,
    "random_state": 0,
}


def _two_regime_series(n_per=150, rng=0):
    rng = np.random.default_rng(rng)
    return np.concatenate([rng.normal(0.0, 0.3, n_per), rng.normal(2.0, 0.3, n_per)])


def test_fit_predict_writes_nothing_to_stdout(capsys):
    labels = E2USDDetector(**SMALL).fit_predict(_two_regime_series())
    assert labels.shape == (300,)
    assert capsys.readouterr().out == ""


def test_predict_writes_nothing_to_stdout(capsys):
    detector = E2USDDetector(**SMALL).fit(_two_regime_series())
    capsys.readouterr()
    detector.predict(_two_regime_series(rng=1))
    assert capsys.readouterr().out == ""
