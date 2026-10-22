"""Tests specific to HidalgoDetector."""

import warnings

import numpy as np
import pytest

from tsseg.algorithms import HidalgoDetector

# the parameters and the series of the docstring example
PARAMS = {"K_states": 2, "burn_in": 0.8, "n_iter": 100, "seed": 10}
FIT_LABELS = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def _two_groups():
    rng = np.random.RandomState(123)
    x = rng.rand(10, 3)
    x[:6, 1:] += 10
    x[6:, 1:] = 0
    return x


def test_predict_on_another_series_of_the_same_length():
    # predict() used to return the labels of the series given to fit
    x1 = _two_groups()
    x2 = x1[::-1].copy()
    expected = HidalgoDetector(**PARAMS).fit_predict(x2)
    assert expected.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    detector = HidalgoDetector(**PARAMS).fit(x1)
    posterior = detector._Pi.copy()
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        assert detector.predict(x2).tolist() == expected.tolist()
    # the fitted attributes still describe x1
    np.testing.assert_array_equal(detector._Pi, posterior)
    assert detector.predict(x1).tolist() == FIT_LABELS


def test_predict_on_a_longer_series_returns_one_label_per_time_point():
    # predict() used to return the 10 labels of the series given to fit
    x1 = _two_groups()
    x_long = np.vstack([x1, x1[:4]])
    expected = HidalgoDetector(**PARAMS).fit_predict(x_long)
    assert len(expected) == 14
    detector = HidalgoDetector(**PARAMS).fit(x1)
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        assert detector.predict(x_long).tolist() == expected.tolist()


def test_predict_after_the_fit_series_is_modified_in_place():
    x1 = _two_groups()
    x = x1.copy()
    detector = HidalgoDetector(**PARAMS).fit(x)
    x[:] = x1[::-1]
    with pytest.warns(UserWarning, match="differs from that given to fit"):
        assert detector.predict(x).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]


def test_predict_on_the_fit_series_reuses_the_fit():
    x1 = _two_groups()
    detector = HidalgoDetector(**PARAMS).fit(x1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # equal values in a new array are the series given to fit
        assert detector.predict(x1.copy()).tolist() == FIT_LABELS
        assert detector.fit_predict(x1).tolist() == FIT_LABELS


def test_points_without_a_probable_state_are_labelled_minus_one():
    # a line and a three-dimensional cloud: intrinsic dimensions 1 and 3
    rng = np.random.default_rng(0)
    line = np.c_[rng.normal(size=30), np.zeros(30), np.zeros(30)]
    cloud = rng.normal(size=(30, 3)) + 5
    x = np.r_[line, cloud]
    detector = HidalgoDetector(K_states=2, n_iter=200, seed=0).fit(x)
    labels = detector.predict(x)
    best = detector._Pi.max(axis=0)
    assert np.any(best < 0.8)
    np.testing.assert_array_equal(labels == -1, best < 0.8)
    np.testing.assert_array_equal(
        labels[best >= 0.8], detector._Pi.argmax(axis=0)[best >= 0.8]
    )
    assert set(labels.tolist()) == {-1, 0, 1}
