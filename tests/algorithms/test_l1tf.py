"""Tests specific to :class:`~tsseg.algorithms.l1tf.TrendFilteringDetector`.

The numerical ground truth is the l1 trend filtering paper (Kim, Koh, Boyd &
Gorinevsky, SIAM Review 51(2):339-360, 2009): the dual box-QP formulation,
its eq. (6) fitting bound ``||y - x||_inf <= 4 lambda``, its eq. (7)
``lambda_max = ||(D D^T)^-1 D y||_inf`` above which the estimate is the best
affine fit, and the affine equivariance of the estimate.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tsseg.algorithms import TrendFilteringDetector
from tsseg.algorithms.l1tf import detector as l1tf


@pytest.fixture
def kinked_signal():
    """Piecewise-linear trend with kinks at 300 and 700 plus small detail."""
    t = np.arange(1000)
    return (
        2.0
        + 0.01 * t
        + 0.05 * np.maximum(t - 300, 0)
        + 0.05 * np.maximum(t - 700, 0)
        + 0.1 * np.sin(t)
    )


def test_recovers_exact_kinks_of_a_clean_ramp():
    # no detail: the kink must be localised to within one sample. The l1
    # optimum may split the second difference over two adjacent rows when
    # its kink falls between two samples, so either 199 or 200 or both may
    # be reported (row t carries the kink at t + 1).
    t = np.arange(500)
    x = 1.0 + 0.02 * t + 0.1 * np.maximum(t - 200, 0)
    detector = TrendFilteringDetector(penalty=10.0)
    change_points = detector.fit_predict(x)
    assert 200 in change_points
    assert set(change_points.tolist()) <= {199, 200, 201}


def test_no_kink_above_lambda_max(kinked_signal):
    detector = TrendFilteringDetector(penalty=10.0)
    detector.fit_predict(kinked_signal)
    lam_max = detector.lambda_max_
    assert lam_max > 100.0  # the demo signal has kinks well below it

    detector = TrendFilteringDetector(penalty=2.0 * lam_max)
    np.testing.assert_array_equal(detector.fit_predict(kinked_signal), [])

    # and eq. (7): the trend is then the best affine fit
    t = np.arange(kinked_signal.size)
    A = np.column_stack([np.ones_like(t, dtype=float), t.astype(float)])
    coef, *_ = np.linalg.lstsq(A, kinked_signal, rcond=None)
    np.testing.assert_allclose(detector.trend_.ravel(), A @ coef, atol=1e-6)


def test_best_fit_bound_eq6():
    # property (6) of the paper: ||y - x||_inf <= 4 * lambda
    rng = np.random.default_rng(0)
    y = rng.standard_normal(200)
    for lam in [0.5, 4.0, 40.0]:
        detector = TrendFilteringDetector(penalty=lam)
        detector.fit_predict(y)
        assert np.abs(y - detector.trend_.ravel()).max() <= 4.0 * lam


def test_second_difference_is_zero_between_kinks(kinked_signal):
    detector = TrendFilteringDetector(penalty=50.0)
    detector.fit_predict(kinked_signal)
    second_diff = detector.second_diff_.ravel()
    # non-kink rows carry an exact zero second difference
    mask = np.ones(second_diff.size, dtype=bool)
    mask[detector.change_points_ - 1] = False
    np.testing.assert_allclose(second_diff[mask], 0.0, atol=1e-8)
    # and every reported kink sits on a genuinely nonzero second difference
    assert np.abs(second_diff[detector.change_points_ - 1]).min() > 1e-6


def test_multivariate_union_of_channel_kinks():
    t = np.arange(400)
    ch1 = 0.02 * t + 0.1 * np.maximum(t - 100, 0)
    ch2 = 1.0 + 0.01 * t + 0.1 * np.maximum(t - 300, 0)
    X = np.column_stack([ch1, ch2])
    detector = TrendFilteringDetector(penalty=10.0)
    change_points = detector.fit_predict(X)
    # kinks localised to within one sample (an l1 kink may fall between
    # two samples and be carried by two adjacent rows)
    assert 100 in change_points and 300 in change_points
    assert set(change_points.tolist()) <= {99, 100, 101, 299, 300, 301}
    assert detector.trend_.shape == (400, 2)
    assert detector.second_diff_.shape == (398, 2)


def test_multivariate_lambda_max_is_channel_maximum():
    t = np.arange(400)
    X = np.column_stack(
        [
            0.02 * t + 0.1 * np.maximum(t - 100, 0),
            1.0 + 0.01 * t + 0.1 * np.maximum(t - 300, 0),
        ]
    )
    detector = TrendFilteringDetector(penalty=10.0)
    detector.fit_predict(X)
    per_channel = [
        TrendFilteringDetector(penalty=10.0).fit_predict(X[:, c])
        for c in range(X.shape[1])
    ]
    # the same penalty detects each channel's kinks in isolation too
    assert 100 in per_channel[0]
    assert set(per_channel[0].tolist()) <= {99, 100, 101}
    assert 300 in per_channel[1]
    assert set(per_channel[1].tolist()) <= {299, 300, 301}


def test_scale_equivariance():
    # x*  is equivariant when the penalty scales with the data: with data
    # a*y + b and penalty a*lambda, the estimate is a*x* + b
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.standard_normal(300)) + 0.02 * np.arange(300)
    lam = 37.0
    is_y = TrendFilteringDetector(penalty=lam)
    cp_y = is_y.fit_predict(y)
    is_scaled = TrendFilteringDetector(penalty=3.0 * lam)
    cp_scaled = is_scaled.fit_predict(3.0 * y + 7.0)
    np.testing.assert_array_equal(cp_y, cp_scaled)
    np.testing.assert_allclose(
        is_scaled.trend_.ravel(), 3.0 * is_y.trend_.ravel() + 7.0, atol=1e-6
    )


def test_constant_series_has_no_kink():
    x = np.full(120, 3.14)
    detector = TrendFilteringDetector(penalty=1.0)
    np.testing.assert_array_equal(detector.fit_predict(x), [])


def test_too_short_series_raises():
    detector = TrendFilteringDetector(penalty=1.0)
    with pytest.raises(ValueError, match="3"):
        detector.fit_predict(np.arange(2.0))


def test_sklearn_contract():
    from sklearn.base import clone

    detector = TrendFilteringDetector(penalty=5.0, knot_tol=1e-4)
    cloned = clone(detector)
    assert cloned.get_params() == detector.get_params()


def test_caps_kkt_certification_escalation(monkeypatch):
    class FakeOSQP:
        def __init__(self):
            self.settings = None
            self.solve_count = 0
            self.warm_starts = []

        def setup(self, **settings):
            self.settings = settings

        def solve(self):
            self.solve_count += 1
            return SimpleNamespace(
                info=SimpleNamespace(status="maximum iterations reached"),
                x=np.array([0.25]),
            )

        def update_settings(self, **settings):
            self.settings.update(settings)

        def warm_start(self, *, x):
            self.warm_starts.append(x.copy())

    solver = FakeOSQP()
    monkeypatch.setitem(
        sys.modules,
        "osqp",
        SimpleNamespace(OSQP=lambda: solver),
    )
    monkeypatch.setattr(
        l1tf,
        "_active_set_polish",
        lambda H, Dy, lam, nu0: (nu0, np.zeros_like(nu0), False),
    )

    with pytest.raises(RuntimeError, match="raise the penalty towards lambda_max_"):
        l1tf._solve_channel(
            np.array([[1.0, -2.0, 1.0]]),
            np.eye(1),
            np.array([0.0, 1.0, 0.0]),
            0.1,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=10,
            polish=True,
        )

    assert solver.solve_count == 2
    assert solver.settings["max_iter"] == 20
    assert solver.settings["eps_abs"] == pytest.approx(1e-9)
    assert solver.settings["eps_rel"] == pytest.approx(1e-9)
    np.testing.assert_array_equal(solver.warm_starts, [[0.1]])
