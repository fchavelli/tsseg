"""CLaP and ClaSP: the parameter schemas accept the options that the code implements.

The schema allowed ``classifier`` in {rocket, knn, svm} and ``merge_score`` in
{cgain, score}, while ``CLaP`` (``clap.py``) implements eight classifiers and
six merge scores. ``classifier="mrhydra"`` was refused by the validation run
in ``fit``, and ``knn``, ``svm`` and ``score`` passed it only to fail inside
``CLaP``. Known change points (``change_points``) failed too: the schema refused
an array and ``CLaP.fit`` reads ``change_points.shape``, so a list failed there.
``ClaspDetector`` refused ``score="f1"``, which ``scoring.py`` implements.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import warnings

import numpy as np
import pytest

pytest.importorskip("numba")

from tsseg.algorithms.clap.clap import CLaP  # noqa: E402
from tsseg.algorithms.clap.clap_detector import ClapDetector  # noqa: E402
from tsseg.algorithms.clap.clasp_detector import ClaspDetector  # noqa: E402
from tsseg.algorithms.clap.distance import _DISTANCE_MAPPING  # noqa: E402
from tsseg.algorithms.clap.scoring import _SCORE_MAPPING  # noqa: E402
from tsseg.algorithms.clap.validation import _VALIDATION_MAPPING  # noqa: E402
from tsseg.algorithms.clap.window_size import _WINDOW_SIZE_MAPPING  # noqa: E402
from tsseg.algorithms.param_schema import (  # noqa: E402
    StrOptions,
    get_parameter_schema,
    validate_params,
)

CLASSIFIERS = [
    "mrhydra",
    "rocket",
    "weasel",
    "quant",
    "rdst",
    "proximityforest",
    "freshprince",
    "inception",
]
MERGE_SCORES = ["cgain", "f1_score", "log_loss", "ami", "hamming_loss", "roc_auc"]


def _compared_strings(func, attribute: str) -> set[str]:
    """Strings compared with ``self.<attribute> == "..."`` in *func*."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
            continue
        left, right = node.left, node.comparators[0]
        if (
            isinstance(node.ops[0], ast.Eq)
            and isinstance(left, ast.Attribute)
            and left.attr == attribute
            and isinstance(right, ast.Constant)
            and isinstance(right.value, str)
        ):
            found.add(right.value)
    return found


@pytest.mark.parametrize(
    ("parameter", "func", "documented"),
    [
        ("classifier", CLaP._cross_val_classifier, CLASSIFIERS),
        ("merge_score", CLaP.fit, MERGE_SCORES),
    ],
)
def test_schema_options_are_the_implemented_branches(parameter, func, documented):
    constraint = get_parameter_schema(ClapDetector)[parameter].constraint
    assert isinstance(constraint, StrOptions)
    implemented = _compared_strings(func, parameter)
    assert implemented == set(documented)
    assert constraint.options == implemented


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("classifier", v) for v in CLASSIFIERS]
    + [("merge_score", v) for v in MERGE_SCORES],
)
def test_implemented_options_pass_validation(parameter, value):
    detector = ClapDetector(**{parameter: value})
    assert validate_params(detector, {"n_samples": 1_000, "n_channels": 1}) == []


@pytest.mark.parametrize(
    ("parameter", "value"),
    [("classifier", "knn"), ("classifier", "svm"), ("merge_score", "score")],
)
def test_unimplemented_options_fail_validation(parameter, value):
    detector = ClapDetector(**{parameter: value}, n_change_points=1)
    with pytest.raises(ValueError, match="parameter validation failed"):
        detector.fit(np.zeros(200))


def _three_change_points():
    rng = np.random.default_rng(0)
    return np.concatenate([rng.normal(m, 1, 150) for m in (0, 4, 0, 4)])


@pytest.mark.parametrize(
    "change_points",
    [[150, 300, 450], (150, 300, 450), np.array([150, 300, 450])],
    ids=["list", "tuple", "array"],
)
def test_known_change_points(change_points):
    pytest.importorskip("aeon")
    detector = ClapDetector(window_size=20, change_points=change_points)
    states = detector.fit_predict(_three_change_points())
    assert states.shape == (600,)
    np.testing.assert_array_equal(detector.change_points_, [150, 300, 450])
    # Each state is constant between the given change points.
    for start, end in [(0, 150), (150, 300), (300, 450), (450, 600)]:
        assert len(np.unique(states[start:end])) == 1


def test_mrhydra_fits():
    # ``classifier="mrhydra"`` is the Fixed configuration of CLaP in TSB-SEG
    # and half of its grid; it failed validation before.
    pytest.importorskip("aeon")
    states = ClapDetector(
        classifier="mrhydra", window_size=20, change_points=[150, 300, 450]
    ).fit_predict(_three_change_points())
    assert states.shape == (600,)
    assert np.issubdtype(states.dtype, np.integer)


def _str_options(detector, parameter: str) -> frozenset[str]:
    constraint = get_parameter_schema(detector)[parameter].constraint
    constraints = constraint if isinstance(constraint, list) else [constraint]
    (options,) = [c.options for c in constraints if isinstance(c, StrOptions)]
    return options


@pytest.mark.parametrize(
    ("detector", "parameter", "mapping"),
    [
        (ClapDetector, "window_size", _WINDOW_SIZE_MAPPING),
        (ClapDetector, "distance", _DISTANCE_MAPPING),
        (ClaspDetector, "window_size", _WINDOW_SIZE_MAPPING),
        (ClaspDetector, "distance", _DISTANCE_MAPPING),
        (ClaspDetector, "score", _SCORE_MAPPING),
        (ClaspDetector, "validation", _VALIDATION_MAPPING),
    ],
)
def test_schema_options_are_the_mapping_keys(detector, parameter, mapping):
    assert _str_options(detector, parameter) == set(mapping)


def test_clasp_f1_score():
    t = np.arange(400)
    x = np.concatenate([np.sin(t / 4), np.sign(np.sin(t / 9))])  # change at 400
    cps = ClaspDetector(
        score="f1", n_change_points=1, window_size=20, n_jobs=1
    ).fit_predict(x)
    assert len(cps) == 1
    assert abs(int(cps[0]) - 400) <= 20


def _mean_shifts():
    """White noise with two mean shifts, at 300 and 700 (the easy series)."""
    rng = np.random.default_rng(42)
    return np.concatenate(
        [
            rng.normal(-0.8, 0.2, 300),
            rng.normal(0.5, 0.25, 400),
            rng.normal(-0.1, 0.1, 300),
        ]
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"n_change_points": 2}, {"n_segments": 3}],
    ids=["n_change_points", "n_segments"],
)
def test_clasp_guided_returns_the_count(kwargs):
    # A given count is not validated: the significance test refused both
    # change points of this series, so the guided mode returned none.
    cps = ClaspDetector(**kwargs, n_jobs=1).fit_predict(_mean_shifts())
    assert len(cps) == 2


def test_clasp_learned_count_is_still_validated():
    # Without a count, the change points still pass the significance test:
    # white noise has none.
    noise = np.random.default_rng(1).normal(0, 1, 600)
    assert ClaspDetector(n_jobs=1).fit_predict(noise).size == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"n_change_points": 2}, {"n_segments": 3}],
    ids=["n_change_points", "n_segments"],
)
def test_clap_guided_uses_the_count(kwargs):
    pytest.importorskip("aeon")
    detector = ClapDetector(**kwargs).fit(_mean_shifts())
    assert len(detector.change_points_) == 2


def test_clap_forwards_the_distance():
    # The z-normalised default finds no change point on this series (its
    # windows have the same shape in every segment); the Euclidean distance
    # sees the mean shifts.
    pytest.importorskip("aeon")
    detector = ClapDetector(distance="euclidean_distance").fit(_mean_shifts())
    assert len(detector.change_points_) == 2
    assert np.all(np.abs(np.asarray(detector.change_points_) - [300, 700]) <= 50)


# predict() on a series other than the fitted one: both detectors returned the
# output of fit whatever X.

GUIDED = {"n_change_points": 2}
REFIT_WARNING = "differs from that given to fit"


def _reversed_mean_shifts():
    """_mean_shifts() reversed: the same length, other change points."""
    return _mean_shifts()[::-1].copy()


def test_clasp_predict_on_another_series_of_the_same_length():
    x1, x2 = _mean_shifts(), _reversed_mean_shifts()
    expected = ClaspDetector(**GUIDED, n_jobs=1).fit_predict(x2)
    detector = ClaspDetector(**GUIDED, n_jobs=1).fit(x1)
    fitted, model = detector.change_points_.copy(), detector.model
    assert not np.array_equal(expected, fitted)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x2), expected)
    # the fitted attributes still describe x1
    assert detector.model is model
    np.testing.assert_array_equal(detector.change_points_, fitted)


def test_clasp_predict_on_a_shorter_series():
    # predict() used to return [206, 637] on a series of 400 points
    x1 = _mean_shifts()
    expected = ClaspDetector(**GUIDED, n_jobs=1).fit_predict(x1[:400])
    detector = ClaspDetector(**GUIDED, n_jobs=1).fit(x1)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        cps = detector.predict(x1[:400])
    np.testing.assert_array_equal(cps, expected)
    assert np.all(cps < 400)


def test_clasp_predict_on_another_series_learns_its_own_count():
    detector = ClaspDetector(distance="euclidean_distance", n_jobs=1)
    detector.fit(_mean_shifts())
    assert len(detector.change_points_) == 2
    # a single regime: no change point passes the significance test
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        assert detector.predict(_mean_shifts()[:300]).size == 0
    assert len(detector.change_points_) == 2


def test_clasp_predict_after_the_fit_series_is_modified_in_place():
    x = _mean_shifts()
    expected = ClaspDetector(**GUIDED, n_jobs=1).fit_predict(_reversed_mean_shifts())
    detector = ClaspDetector(**GUIDED, n_jobs=1).fit(x)
    x[:] = _reversed_mean_shifts()
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x), expected)


def test_clasp_predict_on_the_fit_series_reuses_the_fit():
    x = _mean_shifts()
    detector = ClaspDetector(**GUIDED, n_jobs=1).fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # equal values in a new array are the series given to fit
        assert detector.predict(x.copy()) is detector.change_points_


def test_clap_predict_on_another_series_of_the_same_length():
    pytest.importorskip("aeon")
    x1, x2 = _mean_shifts(), _reversed_mean_shifts()
    expected = ClapDetector(**GUIDED).fit_predict(x2)
    detector = ClapDetector(**GUIDED).fit(x1)
    states, cps = detector.states_.copy(), np.copy(detector.change_points_)
    assert not np.array_equal(expected, states)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x2), expected)
    # the fitted attributes still describe x1
    np.testing.assert_array_equal(detector.states_, states)
    np.testing.assert_array_equal(detector.change_points_, cps)


def test_clap_predict_on_a_shorter_series_returns_one_label_per_time_point():
    # predict() used to return the 1000 labels of the series given to fit
    pytest.importorskip("aeon")
    x1 = _mean_shifts()
    expected = ClapDetector(**GUIDED).fit_predict(x1[:400])
    assert len(expected) == 400
    detector = ClapDetector(**GUIDED).fit(x1)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x1[:400]), expected)


def test_clap_predict_after_the_fit_series_is_modified_in_place():
    pytest.importorskip("aeon")
    x = _mean_shifts()
    expected = ClapDetector(**GUIDED).fit_predict(_reversed_mean_shifts())
    detector = ClapDetector(**GUIDED).fit(x)
    x[:] = _reversed_mean_shifts()
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x), expected)


def test_clap_predict_on_the_fit_series_reuses_the_fit():
    pytest.importorskip("aeon")
    x = _mean_shifts()
    detector = ClapDetector(**GUIDED).fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert detector.predict(x.copy()) is detector.states_
