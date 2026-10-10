"""Unit tests for ``tsseg.metrics.BaseMetric``."""

from __future__ import annotations

import pytest

from tsseg.metrics import BaseMetric


class _EchoMetric(BaseMetric):
    """Returns what it receives, to check the forwarding of ``__call__``."""

    def compute(self, y_true, y_pred, **kwargs):
        return {"score": 0.5, "y_true": y_true, "y_pred": y_pred, "kwargs": kwargs}


def test_base_metric_is_abstract():
    with pytest.raises(TypeError):
        BaseMetric()


def test_keyword_arguments_become_attributes():
    metric = _EchoMetric(margin=3, name="echo")
    assert metric.margin == 3
    assert metric.name == "echo"


def test_call_forwards_to_compute():
    result = _EchoMetric()([1, 2], [3], n_timepoints=10)
    assert result == {
        "score": 0.5,
        "y_true": [1, 2],
        "y_pred": [3],
        "kwargs": {"n_timepoints": 10},
    }
