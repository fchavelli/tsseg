"""Tests for the validation of ``y`` in :meth:`BaseSegmenter.fit`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsseg.algorithms.base import BaseSegmenter


class _RecordingSegmenter(BaseSegmenter):
    """Minimal segmenter whose ``fit`` is not empty, so ``y`` is validated."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
    }

    def __init__(self, *, axis: int = 0):
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        self.y_seen_ = y
        return self

    def _predict(self, X):
        return np.array([], dtype=int)


N_TIMEPOINTS = 20


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(N_TIMEPOINTS, 2))


def _labels():
    return np.repeat([0, 1], N_TIMEPOINTS // 2)


@pytest.mark.parametrize(
    "y",
    [
        _labels(),
        _labels().astype(float),
        pd.Series(_labels()),
        pd.DataFrame({"label": _labels()}),
    ],
    ids=["ndarray-int", "ndarray-float", "series", "dataframe-one-column"],
)
def test_fit_accepts_a_single_numeric_label_series(X, y):
    segmenter = _RecordingSegmenter().fit(X, y)
    assert segmenter.y_seen_ is y


@pytest.mark.parametrize("n_columns", [0, 2, 3])
def test_fit_rejects_a_dataframe_without_exactly_one_column(X, n_columns):
    y = pd.DataFrame({f"label_{i}": _labels() for i in range(n_columns)})
    with pytest.raises(ValueError, match="single column"):
        _RecordingSegmenter().fit(X, y)


def test_fit_predict_rejects_a_two_column_dataframe(X):
    y = pd.DataFrame({"a": _labels(), "b": _labels()})
    with pytest.raises(ValueError, match="single column"):
        _RecordingSegmenter().fit_predict(X, y)


@pytest.mark.parametrize(
    ("y", "message"),
    [
        (np.stack([_labels(), _labels()], axis=1), "should be 1D"),
        (np.array(["a", "b"] * (N_TIMEPOINTS // 2)), "floats or ints"),
        (pd.Series(["a", "b"] * (N_TIMEPOINTS // 2)), "must be numeric"),
        (pd.DataFrame({"label": ["a", "b"] * (N_TIMEPOINTS // 2)}), "must be numeric"),
        (list(_labels()), "Error in input type for y"),
    ],
    ids=[
        "ndarray-2d",
        "ndarray-str",
        "series-str",
        "dataframe-str",
        "list",
    ],
)
def test_fit_rejects_invalid_y(X, y, message):
    with pytest.raises(ValueError, match=message):
        _RecordingSegmenter().fit(X, y)
