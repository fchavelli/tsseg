"""KCPDDetector ``pen_scale``: a penalty expressed as a coefficient on BIC.

The point is transfer: an absolute penalty that suits a 1 000-point univariate
series is wrong for a 50 000-point, 10-channel one, whereas ``log(n) * d`` scales
with both. These tests pin that the scaled penalty is exactly that product, on
the sequence ruptures actually sees, and that the historical absolute behaviour
is untouched -- the merged kcpd rbf campaign depends on it.
"""

import numpy as np
import pytest

from tsseg.algorithms.kcpd.detector import KCPDDetector
from tsseg.algorithms.param_schema import get_parameter_schema, validate_params


def _steps(n_seg=4, seg_len=250, d=2, seed=0):
    """Piecewise-constant means + unit noise, z-normalised per channel."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 4, size=(n_seg, d))
    x = np.concatenate([m + rng.normal(0, 1, size=(seg_len, d)) for m in means])
    x = (x - x.mean(0)) / x.std(0)
    truth = [seg_len * k for k in range(1, n_seg)]
    return x, truth


def test_absolute_penalty_is_unchanged():
    """pen_scale=None must reproduce the historical absolute penalty exactly."""
    x, _ = _steps()
    old = KCPDDetector(pen=5.0, kernel="rbf", decimation=8, min_size=8, jump=4)
    new = KCPDDetector(
        pen=5.0, kernel="rbf", decimation=8, min_size=8, jump=4, pen_scale=None
    )
    np.testing.assert_array_equal(old.fit_predict(x), new.fit_predict(x))


def test_bic_penalty_is_log_n_times_d_on_the_decimated_sequence():
    x, _ = _steps(d=3)
    det = KCPDDetector(pen=1.5, kernel="linear", decimation=8, pen_scale="bic")
    seq = det._decimate(det._ensure_2d(x))
    n, d = seq.shape
    assert n == int(np.ceil(len(x) / 8)) and d == 3
    assert det._resolved_pen(seq) == pytest.approx(1.5 * np.log(n) * d)


def test_bic_scaled_equals_the_same_absolute_penalty_computed_by_hand():
    """Scaling is the only thing pen_scale does: same breakpoints as the explicit value."""
    x, _ = _steps(d=2)
    kw = dict(kernel="linear", decimation=4, min_size=4, jump=2)
    scaled = KCPDDetector(pen=2.0, pen_scale="bic", **kw)
    seq = scaled._decimate(scaled._ensure_2d(x))
    by_hand = KCPDDetector(pen=2.0 * np.log(seq.shape[0]) * seq.shape[1], **kw)
    np.testing.assert_array_equal(scaled.fit_predict(x), by_hand.fit_predict(x))


def test_one_bic_recovers_clean_mean_shifts():
    """Sanity of the scale itself: on unit-variance piecewise-constant data the plain
    BIC coefficient finds the true change points, neither 0 nor dozens."""
    x, truth = _steps(n_seg=4, seg_len=250, d=2)
    cps = KCPDDetector(
        pen=1.0, kernel="linear", pen_scale="bic", min_size=8, jump=4
    ).fit_predict(x)
    assert len(cps) == len(truth)
    assert all(min(abs(c - t) for c in cps) <= 8 for t in truth)


def test_n_cps_ignores_the_penalty_whatever_its_scale():
    x, _ = _steps()
    det = KCPDDetector(n_cps=3, pen=None, kernel="linear", pen_scale="bic")
    assert det._resolved_pen(det._ensure_2d(x)) is None
    assert len(det.fit_predict(x)) == 3


def test_schema_declares_pen_scale_and_rejects_unknown_values():
    assert "pen_scale" in get_parameter_schema(KCPDDetector)
    assert validate_params(KCPDDetector(pen=1.0, pen_scale="bic")) == []
    bad = KCPDDetector(pen=1.0)
    bad.pen_scale = "aic"
    assert any("pen_scale" in e for e in validate_params(bad))
