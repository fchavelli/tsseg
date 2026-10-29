"""Tests of TiccDetector: solver output, number of states chosen by BIC, predict."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsseg.algorithms import TiccDetector
from tsseg.algorithms.ticc.ticc import n_free_parameters


def _series():
    """Three regimes of 200 points: x2 follows x1, then -x1, then x1."""
    rng = np.random.default_rng(0)
    parts = []
    for sign in (1, -1, 1):
        x1 = rng.normal(0, 1, 200)
        x2 = sign * x1 + rng.normal(0, 0.3, 200)
        parts.append(np.column_stack([x1, x2]))
    return np.concatenate(parts)


def _runs(labels):
    """Change points and the label of each segment."""
    cps = np.flatnonzero(np.diff(labels)) + 1
    return cps.tolist(), [int(labels[0])] + [int(labels[c]) for c in cps]


# Outputs of the solver before its refactoring (tsseg 0.3.0) on _series().
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(window_size=3, n_states=2, maxIters=20),
            ([199, 399], [0, 1, 0]),
        ),
        (
            dict(
                window_size=5, n_states=3, lambda_parameter=1e-2, beta=100, maxIters=20
            ),
            ([198, 266, 286, 399], [2, 1, 0, 1, 2]),
        ),
    ],
)
def test_guided_output_is_unchanged(kwargs, expected):
    labels = TiccDetector(**kwargs).fit_predict(_series())

    assert _runs(labels) == expected


def test_bic_selects_the_number_of_states():
    detector = TiccDetector(window_size=3, maxIters=20, max_states=4)

    labels = detector.fit_predict(_series())

    assert detector.n_states_ == 2
    assert sorted(detector.bic_) == [1, 2, 3, 4]
    assert min(detector.bic_, key=detector.bic_.get) == 2
    assert _runs(labels) == ([199, 399], [0, 1, 0])


def test_guided_mode_has_no_bic():
    detector = TiccDetector(window_size=3, n_states=2, maxIters=5)

    detector.fit(_series())

    assert detector.n_states_ == 2
    assert not hasattr(detector, "bic_")


def test_warns_when_bic_selects_max_states():
    detector = TiccDetector(window_size=3, maxIters=20, max_states=2)

    with pytest.warns(UserWarning, match="max_states"):
        detector.fit(_series())

    assert detector.n_states_ == 2


def test_leaves_numpy_global_state_untouched():
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)
    options = np.get_printoptions()

    TiccDetector(window_size=3, n_states=2, maxIters=5).fit(_series())

    assert np.random.random() == expected
    assert np.get_printoptions() == options


def test_n_free_parameters_counts_toeplitz_blocks():
    d, w = 2, 3
    theta = np.zeros((d * w, d * w))
    a0 = np.array([[2.0, 0.5], [0.5, 2.0]])
    a1 = np.array([[0.3, 0.0], [0.1, 0.0]])
    for i in range(w):
        theta[i * d : (i + 1) * d, i * d : (i + 1) * d] = a0
    for i in range(w - 1):
        theta[(i + 1) * d : (i + 2) * d, i * d : (i + 1) * d] = a1
        theta[i * d : (i + 1) * d, (i + 1) * d : (i + 2) * d] = a1.T

    # A0: 3 free entries (symmetric); A1: 2 non-zero entries; A2: none
    assert n_free_parameters(theta, w, d) == 5


def _short_series(n_timepoints):
    return np.random.default_rng(0).normal(size=(n_timepoints, 2))


def _series_with_outlying_end():
    """Noise whose last ``window_size`` rows are outliers, for ``window_size=10``."""
    X = np.random.default_rng(0).normal(size=(600, 2))
    X[-10:] = 1e3
    return X


@pytest.mark.parametrize(
    "X, n_states, match",
    [
        # n_timepoints - window_size + 1 windows, fewer than 2 * n_states
        (_short_series(11), 2, "has only 2 windows"),
        (_short_series(12), 2, "has only 3 windows"),
        (_short_series(11), 3, "has only 2 windows"),
        # enough windows, but the initial mixture gives the outlying ones a state
        (_series_with_outlying_end(), 2, r"gives state 1 only 1 of the 591 windows"),
    ],
)
def test_guided_fit_refuses_a_state_with_fewer_than_two_windows(X, n_states, match):
    detector = TiccDetector(window_size=10, n_states=n_states, maxIters=5)

    with pytest.raises(ValueError, match=match) as excinfo:
        detector.fit(X)

    message = str(excinfo.value)
    assert "at least two windows" in message
    assert "fewer states" in message
    assert "window_size" in message


def test_bic_skips_the_numbers_of_states_with_too_few_windows():
    detector = TiccDetector(window_size=10, maxIters=5, max_states=3)

    labels = detector.fit_predict(_short_series(11))

    assert detector.n_states_ == 1
    assert np.isfinite(detector.bic_[1])
    assert detector.bic_[2] == detector.bic_[3] == np.inf
    assert labels.tolist() == [0] * 11


GUIDED = dict(window_size=3, n_states=2, maxIters=20)
REFIT_WARNING = "differs from that given to fit"


def _swapped_series():
    """_series() with its first two regimes swapped: one change point, at 200."""
    x = _series()
    return np.concatenate([x[200:400], x[:200], x[400:]])


def test_predict_on_another_series_of_the_same_length():
    # predict() used to return the labels of the series given to fit
    x1, x2 = _series(), _swapped_series()
    expected = TiccDetector(**GUIDED).fit_predict(x2)
    assert _runs(expected) == ([199], [1, 0])
    detector = TiccDetector(**GUIDED).fit(x1)
    fitted_model = detector.ticc_
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x2), expected)
    # the fitted attributes still describe x1
    assert detector.ticc_ is fitted_model
    assert _runs(detector.predict(x1)) == ([199, 399], [0, 1, 0])


def test_predict_on_a_shorter_series_returns_one_label_per_time_point():
    # predict() used to return the 600 labels of the series given to fit
    x1 = _series()
    expected = TiccDetector(**GUIDED).fit_predict(x1[:250])
    assert len(expected) == 250
    detector = TiccDetector(**GUIDED).fit(x1)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x1[:250]), expected)


def test_predict_on_another_series_selects_its_own_number_of_states():
    detector = TiccDetector(window_size=3, maxIters=20, max_states=3)
    detector.fit(_series())
    bic = dict(detector.bic_)
    assert detector.n_states_ == 2
    # BIC selects one state on a single regime
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        labels = detector.predict(_series()[:200])
    assert labels.tolist() == [0] * 200
    assert detector.n_states_ == 2
    assert detector.bic_ == bic


def test_predict_after_the_fit_series_is_modified_in_place():
    x = _series()
    detector = TiccDetector(**GUIDED).fit(x)
    x[:] = _swapped_series()
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        assert _runs(detector.predict(x)) == ([199], [1, 0])


def test_predict_on_the_fit_series_reuses_the_fit():
    x1 = _series()
    detector = TiccDetector(**GUIDED).fit(x1)
    labels = detector._state_labels
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # equal values in a new array are the series given to fit
        assert detector.predict(x1.copy()) is labels
        assert _runs(TiccDetector(**GUIDED).fit_predict(x1)) == (
            [199, 399],
            [0, 1, 0],
        )
