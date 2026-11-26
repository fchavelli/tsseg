"""Calibration of the Gaussian likelihood cost (``model="normal"``).

The cost of a segment is n * log det of its maximum likelihood covariance, that
is -2 log-likelihood up to a constant. Rescaling the series moves every segment
cost by n * 2d * log(s), the same total for every partition, so the change
points must not depend on the scale; ``pen_scale="bic"`` must count the d(d+3)/2
parameters of one more segment (d means, d(d+1)/2 covariances); and a segment
of at most d samples, whose covariance is singular, must not be admissible.
"""

from math import log

import numpy as np
import pytest

from tsseg.algorithms.binseg.detector import BinSegDetector
from tsseg.algorithms.bottomup.detector import BottomUpDetector
from tsseg.algorithms.dynp.detector import DynpDetector
from tsseg.algorithms.pelt.detector import PeltDetector
from tsseg.algorithms.ruptures.costs import CostNormal
from tsseg.algorithms.ruptures.detection import Binseg, BottomUp, Dynp, Pelt, Window
from tsseg.algorithms.ruptures.exceptions import NotEnoughPoints
from tsseg.algorithms.ruptures.utils import bic_penalty
from tsseg.algorithms.window.detector import WindowDetector

SCALES = (1.0, 1e-3, 1e3)


def _covariance_changes(seed=0, length=80):
    """Three channels, zero mean, two changes of covariance at 80 and 160."""
    rng = np.random.default_rng(seed)
    covariances = [
        np.eye(3),
        np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.diag([4.0, 1.0, 0.25]),
    ]
    return np.vstack(
        [rng.multivariate_normal(np.zeros(3), c, length) for c in covariances]
    )


def _penalised(name, pen, pen_scale=None):
    common = {"model": "normal", "min_size": 10, "jump": 5, "pen_scale": pen_scale}
    if name == "pelt":
        return PeltDetector(penalty=pen, **common)
    if name == "binseg":
        return BinSegDetector(penalty=pen, **common)
    if name == "bottomup":
        return BottomUpDetector(penalty=pen, **common)
    return WindowDetector(width=60, pen=pen, **common)


@pytest.mark.parametrize("name", ["pelt", "binseg", "bottomup"])
@pytest.mark.parametrize("pen_scale", [None, "bic"])
def test_change_points_do_not_depend_on_the_scale(name, pen_scale):
    x = _covariance_changes()
    pen = 1.0 if pen_scale == "bic" else 9 * log(len(x))
    found = [
        _penalised(name, pen, pen_scale).fit_predict(s * x).tolist() for s in SCALES
    ]
    assert found[1:] == found[:1] * 2
    if name == "pelt":  # exact; the greedy searches land within 10 samples
        assert found[0] == [80, 160]
    assert len(found[0]) == 2
    assert np.abs(np.array(found[0]) - [80, 160]).max() <= 10


def test_window_does_not_depend_on_the_scale():
    x = _covariance_changes()
    found = [
        _penalised("window", 1.0, "bic").fit_predict(s * x).tolist() for s in SCALES
    ]
    assert found[1:] == found[:1] * 2
    assert len(found[0]) == 2


def test_dynp_does_not_depend_on_the_scale():
    x = _covariance_changes()
    found = [
        DynpDetector(model="normal", n_cps=2, min_size=10, jump=5)
        .fit_predict(s * x)
        .tolist()
        for s in SCALES
    ]
    assert found == [[80, 160]] * len(SCALES)


@pytest.mark.parametrize("d", [1, 4, 8])
def test_bic_does_not_cut_white_noise(d):
    rng = np.random.default_rng(d)
    x = rng.standard_normal((1200, d))
    x = (x - x.mean(0)) / x.std(0)
    detector = PeltDetector(model="normal", penalty=1.0, pen_scale="bic", jump=5)
    assert detector.fit_predict(x).tolist() == []


@pytest.mark.parametrize("d", [1, 3, 8])
def test_bic_unit_is_the_parameter_count(d):
    x = np.random.default_rng(0).standard_normal((500, d))
    for scale in SCALES:
        cost = CostNormal().fit(scale * x)
        assert bic_penalty(2.0, cost) == pytest.approx(2.0 * log(500) * d * (d + 3) / 2)


def test_segment_cost_is_the_maximum_likelihood_log_determinant():
    rng = np.random.default_rng(1)
    for d in (1, 2, 5):
        x = rng.standard_normal((200, d)) @ rng.standard_normal((d, d))
        cost = CostNormal().fit(x)
        ridge = 1e-6 * np.var(x, axis=0).mean()
        segment = x[30:90]
        covariance = np.atleast_2d(np.cov(segment.T, bias=True)) + ridge * np.eye(d)
        expected = 60 * np.linalg.slogdet(covariance)[1]
        assert cost.error(30, 90) == pytest.approx(expected, rel=1e-12)
        unregularised = CostNormal(add_small_diag=False).fit(x)
        covariance = np.atleast_2d(np.cov(segment.T, bias=True))
        expected = 60 * np.linalg.slogdet(covariance)[1]
        assert unregularised.error(30, 90) == pytest.approx(expected, rel=1e-12)


def test_constant_signal_keeps_a_finite_cost():
    cost = CostNormal().fit(np.full((50, 2), 3.0))
    assert np.isfinite(cost.error(0, 50))
    detector = PeltDetector(model="normal", penalty=1.0, pen_scale="bic")
    assert detector.fit_predict(np.full((50, 2), 3.0)).tolist() == []


def test_min_size_exceeds_the_dimension():
    rng = np.random.default_rng(2)
    cost = CostNormal()
    assert cost.fit(rng.standard_normal((100, 8))).min_size == 9
    assert cost.fit(rng.standard_normal(100)).min_size == 2
    for search in (Pelt, Dynp, Binseg, BottomUp):
        algo = search(model="normal", min_size=2, jump=1)
        assert algo.fit(rng.standard_normal((100, 8))).min_size == 9
        assert algo.fit(rng.standard_normal((100, 1))).min_size == 2
        assert (
            search(model="normal", min_size=12)
            .fit(rng.standard_normal((100, 8)))
            .min_size
            == 12
        )


def test_segments_are_longer_than_the_dimension():
    d = 8
    x = np.random.default_rng(3).standard_normal((600, d))
    cps = PeltDetector(
        model="normal", penalty=d * log(600), min_size=2, jump=1
    ).fit_predict(x)
    lengths = np.diff(np.concatenate([[0], cps, [600]]))
    assert lengths.min() >= d + 1


def test_window_halves_must_exceed_the_dimension():
    x = np.random.default_rng(4).standard_normal((200, 8))
    with pytest.raises(NotEnoughPoints):
        Window(width=16, model="normal").fit(x)
    Window(width=18, model="normal").fit(x)
