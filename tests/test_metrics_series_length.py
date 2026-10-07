"""Series length of the change-point metrics (``n_timepoints``).

Without ``n_timepoints``, ``F1Score``, ``Covering``, ``BidirectionalCovering``
and ``GaussianF1Score`` read the series length from the change points
themselves, so ``y_true`` must end with it. The raw output of a detector does
not: its last true change point was then taken for the length and ignored,
without any error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsseg.metrics import BidirectionalCovering, Covering, F1Score, GaussianF1Score

# Series of length 1000 with true change points 300 and 700; one predicted
# change point at 300. Expected scores computed by hand.
N = 1000
TRUE_CPS = [300, 700]
PRED_CPS = [300]
EXPECTED = {
    "F1Score": 2 / 3,  # precision 1, recall 1/2
    "Covering": (300 + 400 * 400 / 700 + 300 * 300 / 700) / N,
    "BidirectionalCovering": None,  # harmonic mean, checked below
    "GaussianF1Score": 2 / 3,
}

METRICS = [
    pytest.param(lambda: F1Score(margin=5), id="F1Score"),
    pytest.param(Covering, id="Covering"),
    pytest.param(BidirectionalCovering, id="BidirectionalCovering"),
    pytest.param(GaussianF1Score, id="GaussianF1Score"),
]

_INFERRED = "series length is inferred"


def _expected(metric) -> float:
    name = type(metric).__name__
    if name == "BidirectionalCovering":
        gt = EXPECTED["Covering"]
        pred = (300 + 700 * 400 / 700) / N
        return 2 * gt * pred / (gt + pred)
    return EXPECTED[name]


@pytest.mark.parametrize("make_metric", METRICS)
def test_raw_detector_output_with_n_timepoints(make_metric):
    metric = make_metric()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = metric.compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
    assert result["score"] == pytest.approx(_expected(metric))


@pytest.mark.parametrize("make_metric", METRICS)
def test_boundaries_optional_with_n_timepoints(make_metric):
    metric = make_metric()
    bare = metric.compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
    with_bounds = metric.compute([0, *TRUE_CPS, N], [0, *PRED_CPS, N], n_timepoints=N)
    as_arrays = metric.compute(
        np.array(TRUE_CPS), np.array(PRED_CPS, dtype=int), n_timepoints=N
    )
    assert bare == with_bounds == as_arrays


@pytest.mark.parametrize("make_metric", METRICS)
def test_legacy_convention_unchanged_but_warns(make_metric):
    """``y_true`` ending with n gives the same scores as ``n_timepoints``."""
    metric = make_metric()
    with pytest.warns(UserWarning, match=_INFERRED):
        legacy = metric.compute([*TRUE_CPS, N], PRED_CPS)
    assert legacy == metric.compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
    assert legacy["score"] == pytest.approx(_expected(metric))


@pytest.mark.parametrize("make_metric", METRICS)
def test_legacy_pitfall_is_flagged(make_metric):
    """The pitfall itself is kept (same score as before) but now warns."""
    metric = make_metric()
    with pytest.warns(UserWarning, match=_INFERRED):
        result = metric.compute(TRUE_CPS, PRED_CPS)
    assert result["score"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "make_metric",
    [
        pytest.param(lambda: F1Score(margin=5, convert_labels_to_segments=True)),
        pytest.param(lambda: Covering(convert_labels_to_segments=True)),
        pytest.param(lambda: BidirectionalCovering(convert_labels_to_segments=True)),
        pytest.param(lambda: GaussianF1Score(convert_labels_to_segments=True)),
    ],
    ids=["F1Score", "Covering", "BidirectionalCovering", "GaussianF1Score"],
)
def test_labels_need_no_n_timepoints(make_metric):
    metric = make_metric()
    y_true = np.zeros(N, dtype=int)
    y_true[300:700] = 1
    y_true[700:] = 2
    y_pred = np.zeros(N, dtype=int)
    y_pred[300:] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from_labels = metric.compute(y_true, y_pred)
        with_n = metric.compute(y_true, y_pred, n_timepoints=N)
    assert from_labels == with_n
    assert from_labels["score"] == pytest.approx(_expected(metric))
    with pytest.raises(ValueError, match="n_timepoints"):
        metric.compute(y_true, y_pred, n_timepoints=N + 1)


@pytest.mark.parametrize("make_metric", METRICS)
def test_no_change_point_at_all(make_metric):
    metric = make_metric()
    assert metric.compute([], [], n_timepoints=N)["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("make_metric", METRICS)
def test_change_point_outside_series_rejected(make_metric):
    metric = make_metric()
    with pytest.raises(ValueError, match="n_timepoints"):
        metric.compute([300, 1200], PRED_CPS, n_timepoints=N)
    with pytest.raises(ValueError, match="n_timepoints"):
        metric.compute(TRUE_CPS, [-1], n_timepoints=N)


def test_f1_missed_and_spurious_change_points():
    f1 = F1Score(margin=5)
    assert f1.compute([500], [], n_timepoints=N)["score"] == 0.0
    assert f1.compute([], [500], n_timepoints=N)["score"] == 0.0
    # A float margin is a fraction of n_timepoints: 1 % of 1000 = 10 points.
    assert F1Score(margin=0.01).compute([500], [509], n_timepoints=N)[
        "score"
    ] == pytest.approx(1.0)
    assert F1Score(margin=0.01).compute([500], [511], n_timepoints=N)["score"] == 0.0
