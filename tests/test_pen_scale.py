"""``pen_scale="bic"`` of the penalised ruptures detectors: the penalty is a
coefficient on ``log(n) * u``, u the cost of one sample of the whole signal
(``ruptures.utils.tie_unit``)."""

import numpy as np
import pytest

from tsseg.algorithms.binseg.detector import BinSegDetector
from tsseg.algorithms.bottomup.detector import BottomUpDetector
from tsseg.algorithms.pelt.detector import PeltDetector
from tsseg.algorithms.ruptures.costs import cost_factory
from tsseg.algorithms.ruptures.utils import bic_penalty, tie_unit
from tsseg.algorithms.window.detector import WindowDetector

MODELS = ("l1", "l2", "rbf", "cosine")


def _signal(seed=0, n=300, d=3):
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [rng.normal(rng.normal(0, 2, d), 1.0, (length, d)) for length in (80, 120, 100)]
    )
    return (x - x.mean(0)) / x.std(0)


def _detector(name, model, pen, pen_scale=None):
    common = {"model": model, "min_size": 2, "jump": 5, "pen_scale": pen_scale}
    if name == "pelt":
        return PeltDetector(penalty=pen, **common)
    if name == "binseg":
        return BinSegDetector(penalty=pen, **common)
    if name == "bottomup":
        return BottomUpDetector(penalty=pen, **common)
    return WindowDetector(width=40, pen=pen, **common)


@pytest.mark.parametrize("name", ["pelt", "binseg", "bottomup", "window"])
@pytest.mark.parametrize("model", MODELS)
def test_bic_is_the_scaled_penalty(name, model):
    x = _signal()
    coefficient = 0.7
    pen = bic_penalty(coefficient, cost_factory(model).fit(x))
    scaled = _detector(name, model, coefficient, "bic").fit_predict(x)
    absolute = _detector(name, model, pen).fit_predict(x)
    assert np.array_equal(scaled, absolute)


def test_bic_unit():
    x = _signal(d=4)
    n = x.shape[0]
    # unit-variance channels: the l2 unit is d, the textbook BIC log(n) * d
    assert bic_penalty(1.0, cost_factory("l2").fit(x)) == pytest.approx(np.log(n) * 4)
    for model in ("rbf", "cosine"):
        assert bic_penalty(1.0, cost_factory(model).fit(x)) == pytest.approx(np.log(n))
    l1 = cost_factory("l1").fit(x)
    assert bic_penalty(1.0, l1) == pytest.approx(np.log(n) * tie_unit(l1))
    # a constant signal costs 0 everywhere: unit 1 (l2: the floor of tie_unit)
    assert bic_penalty(2.0, cost_factory("l1").fit(np.ones((50, 2)))) == pytest.approx(
        2.0 * np.log(50)
    )


@pytest.mark.parametrize("name", ["pelt", "binseg", "bottomup", "window"])
@pytest.mark.parametrize("model", MODELS)
def test_bic_keeps_a_constant_signal_whole(name, model):
    for level in (0.0, 1.0, 1e8 + 0.1234567):
        x = np.full((200, 2), level)
        assert _detector(name, model, 0.01, "bic").fit_predict(x).size == 0


@pytest.mark.parametrize("model", MODELS)
def test_bic_does_not_depend_on_the_units_of_the_signal(model):
    x = _signal(1)
    expected = PeltDetector(model=model, penalty=1.0, pen_scale="bic").fit_predict(x)
    for scale in (1e-3, 2.0**10):
        found = PeltDetector(model=model, penalty=1.0, pen_scale="bic").fit_predict(
            x * scale
        )
        assert np.array_equal(found, expected)


def test_n_cps_overrides_the_scaled_penalty():
    x = _signal(2)
    with pytest.warns(UserWarning):
        det = BinSegDetector(n_cps=2, penalty=0.5, pen_scale="bic")
    assert det.fit_predict(x).size == 2


@pytest.mark.parametrize("cls", [BinSegDetector, BottomUpDetector, WindowDetector])
@pytest.mark.parametrize("model", MODELS)
def test_n_cps_zero_keeps_the_signal_whole(cls, model):
    # guided runs on a series without change points (some of HAS and TSSB)
    det = cls(n_cps=0, model=model, min_size=2, jump=5)
    assert det.fit_predict(_signal(3)).size == 0


def test_unknown_pen_scale():
    with pytest.raises(ValueError, match="pen_scale"):
        PeltDetector(penalty=1.0, pen_scale="aic")
    with pytest.raises(ValueError, match="pen_scale"):
        WindowDetector(pen=1.0, pen_scale="log")
