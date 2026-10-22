"""Stopping criteria of ``BinSegDetector`` and ``BottomUpDetector``.

``penalty`` defaults to ``None``, which means a penalty of 10 when neither
``n_cps`` nor ``epsilon`` is given: guiding the detector by ``n_cps`` alone, or
stopping it by ``epsilon`` alone, then needs no ``penalty=None``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsseg.algorithms.binseg.detector import BinSegDetector
from tsseg.algorithms.bottomup.detector import BottomUpDetector

DETECTORS = [BinSegDetector, BottomUpDetector]


def _signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    means = np.repeat([0.0, 4.0, -3.0, 2.0], 100)
    return (means + 0.5 * rng.standard_normal(means.size))[:, None]


@pytest.mark.parametrize("cls", DETECTORS)
def test_n_cps_alone_does_not_warn(cls):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        det = cls(n_cps=3)
        cps = det.fit_predict(_signal())
    assert det.penalty is None
    assert cps.size == 3


@pytest.mark.parametrize("cls", DETECTORS)
def test_n_cps_with_an_explicit_penalty_still_warns(cls):
    with pytest.warns(UserWarning, match="penalty is ignored"):
        det = cls(n_cps=3, penalty=5.0)
    assert det.penalty is None
    assert det.fit_predict(_signal()).size == 3


@pytest.mark.parametrize("cls", DETECTORS)
def test_default_penalty_is_ten(cls):
    x = _signal()
    det = cls()
    assert det.penalty == 10.0
    np.testing.assert_array_equal(det.fit_predict(x), cls(penalty=10).fit_predict(x))


@pytest.mark.parametrize("cls", DETECTORS)
def test_penalty_none_alone_falls_back_to_ten(cls):
    x = _signal()
    det = cls(penalty=None)
    assert det.penalty == 10.0
    np.testing.assert_array_equal(det.fit_predict(x), cls().fit_predict(x))


@pytest.mark.parametrize("cls", DETECTORS)
def test_epsilon_alone_is_accepted(cls):
    x = _signal()
    # total cost of the 4-segment truth: 400 points of variance 0.25
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        det = cls(epsilon=150.0)
    assert det.penalty is None
    assert det.epsilon == 150.0
    np.testing.assert_array_equal(
        det.fit_predict(x), cls(penalty=None, epsilon=150.0).fit_predict(x)
    )


@pytest.mark.parametrize("cls", DETECTORS)
def test_penalty_and_epsilon_together_are_refused(cls):
    with pytest.raises(ValueError, match="at most one"):
        cls(penalty=5.0, epsilon=150.0)


@pytest.mark.parametrize("cls", DETECTORS)
@pytest.mark.parametrize(
    "params", [{}, {"n_cps": 2}, {"epsilon": 150.0}, {"penalty": 3.0}]
)
def test_clone_keeps_the_stopping_criterion(cls, params):
    det = cls(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        copy = det.clone()
    assert copy.get_params() == det.get_params()
