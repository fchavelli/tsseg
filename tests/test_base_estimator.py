"""Unit tests for the estimator base classes of :mod:`tsseg.algorithms.base`.

The generic suite (``tests/algorithms``) runs every detector through ``fit``
and ``predict``; these tests pin the base-class machinery itself on minimal
segmenters: tags, fitted parameters, ``reset``, ``clone``, input checks and
conversions, the parameter-schema hook and the label converters.
``_check_y`` has its own file (``test_base_check_y.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from tsseg.algorithms.base import BaseSegmenter, BaseSeriesEstimator, _infer_inner_type
from tsseg.algorithms.param_schema import Closed, DataDependent, Interval, ParamDef

N_TIMEPOINTS = 20


class _Segmenter(BaseSegmenter):
    """Segmenter with a non-empty fit that records what it receives."""

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": True,
        "detector_type": "change_point_detection",
    }
    _parameter_schema = {
        "window": ParamDef(constraint=Interval(int, 1, None, Closed.LEFT)),
        "_cross_constraints": [
            DataDependent("window < n_samples", "window must fit in the series")
        ],
    }

    def __init__(
        self,
        *,
        window: int = 2,
        config: dict | None = None,
        random_state: int | None = None,
        axis: int = 0,
    ):
        self.window = window
        self.config = config
        self.random_state = random_state
        super().__init__(axis=axis)

    def _fit(self, X, y=None):
        self.X_fit_ = X
        self.scaler_ = StandardScaler().fit(np.asarray(X))
        return self

    def _predict(self, X):
        self.X_predict_ = X
        return np.array([X.shape[0] // 2])


class _EmptyFitSegmenter(_Segmenter):
    _tags = {"fit_is_empty": True}


class _UnivariateSegmenter(_Segmenter):
    _tags = {"capability:multivariate": False}


class _MultivariateOnlySegmenter(_Segmenter):
    _tags = {"capability:univariate": False}


class _MissingValuesSegmenter(_Segmenter):
    _tags = {"capability:missing_values": True}


class _DataFrameSegmenter(_Segmenter):
    _tags = {"X_inner_type": "pd.DataFrame"}


class _ListInnerTypeSegmenter(_Segmenter):
    _tags = {"X_inner_type": ["pd.DataFrame", "np.ndarray"]}


class _RequiresYSegmenter(_Segmenter):
    _tags = {"requires_y": True}


class _DefaultFitSegmenter(BaseSegmenter):
    """Non-empty fit that relies on the inherited ``_fit``."""

    _tags = {"fit_is_empty": False}

    def __init__(self, *, axis: int = 0):
        super().__init__(axis=axis)

    def _predict(self, X):
        return np.array([], dtype=int)


class _DynamicTagSegmenter(_DefaultFitSegmenter):
    """Sets a tag from a hyperparameter in ``__init__``."""

    def __init__(self, *, multivariate: bool = False, axis: int = 0):
        self.multivariate = multivariate
        super().__init__(axis=axis)
        self.set_tags(**{"capability:multivariate": multivariate})


class _BrokenPropertySegmenter(_Segmenter):
    @property
    def broken_(self):
        raise AttributeError("not available")


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(N_TIMEPOINTS, 3))


# ---------------------------------------------------------------------------
# Inner type names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("ndarray", "np.ndarray"),
        ("numpy", "np.ndarray"),
        ("DataFrame", "pd.DataFrame"),
        ("Series", "pd.Series"),
        ("list", "list"),
    ],
)
def test_infer_inner_type(name, expected):
    assert _infer_inner_type(name) == expected


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def test_class_tags_are_merged_along_the_mro():
    tags = _UnivariateSegmenter.get_class_tags()
    assert tags["cant_pickle"] is False  # BaseAeonEstimator
    assert tags["X_inner_type"] == "np.ndarray"  # BaseSeriesEstimator
    assert tags["requires_y"] is False  # BaseSegmenter
    assert tags["fit_is_empty"] is False  # _Segmenter overrides BaseSegmenter
    assert tags["capability:multivariate"] is False  # most derived class wins


def test_get_class_tags_returns_a_copy():
    _Segmenter.get_class_tags()["fit_is_empty"] = True
    assert _Segmenter.get_class_tag("fit_is_empty") is False


def test_get_class_tag_missing_name():
    with pytest.raises(ValueError, match="could not be found"):
        _Segmenter.get_class_tag("no_such_tag")
    assert _Segmenter.get_class_tag("no_such_tag", raise_error=False) is None
    assert (
        _Segmenter.get_class_tag(
            "no_such_tag", raise_error=False, tag_value_default="fallback"
        )
        == "fallback"
    )


def test_set_tags_overrides_the_instance_only():
    est, other = _Segmenter(), _Segmenter()
    assert est.set_tags(**{"capability:multivariate": False}) is est
    assert est.get_tag("capability:multivariate") is False
    assert other.get_tag("capability:multivariate") is True
    assert _Segmenter.get_class_tag("capability:multivariate") is True


def test_set_tags_copies_the_values():
    est = _Segmenter()
    dependencies = ["numpy"]
    est.set_tags(python_dependencies=dependencies)
    dependencies.append("torch")
    assert est.get_tag("python_dependencies") == ["numpy"]


def test_get_tag_missing_name():
    est = _Segmenter()
    with pytest.raises(ValueError, match="could not be found"):
        est.get_tag("no_such_tag")
    assert est.get_tag("no_such_tag", raise_error=False, tag_value_default=3) == 3


@pytest.mark.parametrize("multivariate", [True, False])
def test_tags_set_in_init_survive_the_reset_of_fit(X, multivariate):
    est = _DynamicTagSegmenter(multivariate=multivariate)
    if multivariate:
        est.fit(X)
        assert est.get_tag("capability:multivariate") is True
    else:
        with pytest.raises(ValueError, match="Multivariate data not supported"):
            est.fit(X)


# ---------------------------------------------------------------------------
# Fitted parameters
# ---------------------------------------------------------------------------


def test_get_fitted_params_requires_fit():
    with pytest.raises(NotFittedError, match="has not been fitted yet"):
        _Segmenter().get_fitted_params()


def test_get_fitted_params_lists_public_trailing_underscore_attributes(X):
    params = _Segmenter(window=3).fit(X).get_fitted_params(deep=False)
    assert set(params) == {"X_fit_", "scaler_", "metadata_"}
    np.testing.assert_array_equal(params["X_fit_"], X)
    assert params["metadata_"]["n_channels"] == 3


def test_get_fitted_params_deep_flattens_nested_estimators(X):
    params = _Segmenter().fit(X).get_fitted_params(deep=True)
    np.testing.assert_allclose(params["scaler___mean_"], X.mean(axis=0))
    assert params["scaler___n_features_in_"] == 3
    shallow = _Segmenter().fit(X).get_fitted_params(deep=False)
    assert not any("__" in key for key in shallow)


def test_get_fitted_params_skips_attributes_that_raise(X):
    params = _BrokenPropertySegmenter().fit(X).get_fitted_params()
    assert "broken_" not in params
    assert "X_fit_" in params


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_clears_the_fitted_state_and_keeps_the_parameters(X):
    est = _Segmenter().set_params(window=4)
    est.fit(X)
    assert est.reset() is est
    assert not est.is_fitted
    assert not hasattr(est, "X_fit_")
    assert est.metadata_ == {}
    assert est.get_params(deep=False)["window"] == 4


@pytest.mark.parametrize(
    "keep, kept",
    [("X_fit_", {"X_fit_"}), (["X_fit_", "scaler_"], {"X_fit_", "scaler_"})],
    ids=["str", "list"],
)
def test_reset_keep(X, keep, kept):
    est = _Segmenter().fit(X)
    est.reset(keep=keep)
    for attr in ("X_fit_", "scaler_"):
        assert hasattr(est, attr) is (attr in kept)
    assert not est.is_fitted


def test_reset_keeps_double_underscore_attributes(X):
    est = _Segmenter().fit(X)
    est.cache__entry = "kept"
    est.reset()
    assert est.cache__entry == "kept"


def test_reset_keep_cannot_protect_attributes_set_by_init(X):
    """``__init__`` runs again after the deletion, as in aeon."""
    est = _Segmenter().fit(X)
    est.reset(keep="metadata_")
    assert est.metadata_ == {}


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------


def test_clone_of_a_fitted_estimator_is_unfitted(X):
    est = _Segmenter(window=5, random_state=3).fit(X)
    cloned = est.clone()
    assert type(cloned) is _Segmenter
    assert cloned is not est
    assert cloned.get_params(deep=False) == est.get_params(deep=False)
    assert not cloned.is_fitted
    assert not hasattr(cloned, "X_fit_")


def test_clone_without_random_state_keeps_it():
    assert _Segmenter(random_state=3).clone().random_state == 3


@pytest.mark.xfail(
    strict=True,
    reason="T69: clone(random_state=...) only calls a set_random_state method "
    "that no detector defines, so the seed is silently ignored (aeon sets every "
    "random_state parameter of the clone)",
)
def test_clone_with_random_state_seeds_the_clone():
    first = _Segmenter().clone(random_state=7)
    second = _Segmenter().clone(random_state=7)
    assert first.random_state is not None
    assert first.random_state == second.random_state


@pytest.mark.xfail(
    strict=True,
    reason="T70: clone() passes the parameter objects themselves to the new "
    "instance, so a mutable parameter is shared with the original (sklearn's "
    "clone, used by aeon, deep-copies them)",
)
def test_clone_does_not_share_mutable_parameters():
    est = _Segmenter(config={"lags": [1]})
    cloned = est.clone()
    cloned.config["lags"].append(2)
    assert est.config == {"lags": [1]}


# ---------------------------------------------------------------------------
# axis and input checks
# ---------------------------------------------------------------------------


def test_constructor_rejects_an_invalid_axis():
    with pytest.raises(ValueError, match="axis should be 0 or 1"):
        _Segmenter(axis=2)


def test_fit_rejects_an_invalid_axis(X):
    with pytest.raises(ValueError, match="Input axis should be 0 or 1, saw 2"):
        _Segmenter().fit(X, axis=2)


@pytest.mark.parametrize(
    "bad_X, message",
    [
        ([1.0, 2.0, 3.0], "Input type of X should be one of"),
        (np.array(["a", "b", "c"], dtype=object), "dtype for np.ndarray"),
        (np.zeros(5, dtype=bool), "dtype for np.ndarray"),
        (pd.Series(["a", "b", "c"]), "pd.Series dtype must be numeric"),
        (
            pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]}),
            "pd.DataFrame dtype must be numeric",
        ),
        (np.zeros((4, 3, 2)), "at most 2 dimensions"),
    ],
    ids=["list", "object-array", "bool-array", "str-series", "str-column", "3d"],
)
def test_fit_rejects_unsupported_inputs(bad_X, message):
    with pytest.raises(ValueError, match=message):
        _Segmenter().fit(bad_X)


@pytest.mark.parametrize(
    "with_nan",
    [
        np.array([0.0, np.nan, 1.0, 2.0, 3.0]),
        pd.Series([0.0, np.nan, 1.0, 2.0, 3.0]),
        pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0], "b": [0.0, np.nan, 1, 2, 3]}),
    ],
    ids=["ndarray", "series", "dataframe"],
)
def test_missing_values_follow_the_tag(with_nan):
    with pytest.raises(ValueError, match="Missing values not supported"):
        _Segmenter().fit(with_nan)
    est = _MissingValuesSegmenter().fit(with_nan)
    assert est.metadata_["missing_values"]


def test_univariate_only_detector(X):
    with pytest.raises(ValueError, match="Multivariate data not supported"):
        _UnivariateSegmenter().fit(X)
    est = _UnivariateSegmenter().fit(X[:, :1])  # one channel is univariate
    assert est.metadata_ == {
        "multivariate": False,
        "n_channels": 1,
        "missing_values": False,
    }


def test_multivariate_only_detector(X):
    with pytest.raises(ValueError, match="Univariate data not supported"):
        _MultivariateOnlySegmenter().fit(X[:, 0])
    assert _MultivariateOnlySegmenter().fit(X).metadata_["multivariate"]


@pytest.mark.parametrize(
    "shape, axis, n_channels",
    [((N_TIMEPOINTS,), 0, 1), ((N_TIMEPOINTS, 3), 0, 3), ((3, N_TIMEPOINTS), 1, 3)],
)
def test_metadata_counts_the_channels_along_the_axis(shape, axis, n_channels):
    est = _Segmenter().fit(np.ones(shape), axis=axis)
    assert est.metadata_["n_channels"] == n_channels
    assert est.metadata_["multivariate"] is (n_channels > 1)
    assert est._data_context == {"n_samples": N_TIMEPOINTS, "n_channels": n_channels}


def test_build_data_context_falls_back_to_len():
    ctx = BaseSeriesEstimator._build_data_context([1, 2, 3, 4], 0, {})
    assert ctx == {"n_samples": 4, "n_channels": 1}


# ---------------------------------------------------------------------------
# Conversion to the inner type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_X",
    [
        lambda X: X,
        lambda X: pd.DataFrame(X, columns=["a", "b", "c"]),
    ],
    ids=["ndarray", "dataframe"],
)
def test_two_dimensional_inputs_reach_fit_as_an_array(X, make_X):
    est = _Segmenter().fit(make_X(X))
    assert isinstance(est.X_fit_, np.ndarray)
    np.testing.assert_array_equal(est.X_fit_, X)


@pytest.mark.parametrize("make_x", [np.asarray, pd.Series], ids=["ndarray", "series"])
@pytest.mark.parametrize(
    "axis, shape", [(0, (N_TIMEPOINTS, 1)), (1, (1, N_TIMEPOINTS))]
)
def test_univariate_inputs_get_a_channel_dimension(make_x, axis, shape):
    x = np.arange(N_TIMEPOINTS, dtype=float)
    est = _Segmenter(axis=axis).fit(make_x(x))
    assert est.X_fit_.shape == shape
    np.testing.assert_array_equal(est.X_fit_.ravel(), x)


@pytest.mark.parametrize(
    "make_X", [np.asarray, pd.DataFrame], ids=["ndarray", "dataframe"]
)
def test_fit_transposes_an_input_given_along_the_other_axis(X, make_X):
    # window=10 passes the schema only if n_samples is read along the time axis.
    est = _Segmenter(window=10).fit(make_X(X.T), axis=1)
    np.testing.assert_array_equal(np.asarray(est.X_fit_), X)
    assert est._data_context["n_samples"] == N_TIMEPOINTS


@pytest.mark.parametrize(
    "make_X, axis, expected_shape",
    [
        (lambda X: X, 0, (N_TIMEPOINTS, 3)),
        (lambda X: pd.DataFrame(X), 0, (N_TIMEPOINTS, 3)),
        (lambda X: pd.Series(X[:, 0]), 0, (N_TIMEPOINTS, 1)),
        (lambda X: pd.DataFrame(X.T), 1, (N_TIMEPOINTS, 3)),
    ],
    ids=["ndarray", "dataframe", "series", "dataframe-axis1"],
)
@pytest.mark.parametrize("cls", [_DataFrameSegmenter, _ListInnerTypeSegmenter])
def test_dataframe_inner_type(X, cls, make_X, axis, expected_shape):
    est = cls().fit(make_X(X), axis=axis)
    assert isinstance(est.X_fit_, pd.DataFrame)
    assert est.X_fit_.shape == expected_shape
    np.testing.assert_array_equal(est.X_fit_.to_numpy()[:, 0], X[:, 0])


@pytest.mark.parametrize(
    "X_in, axis, message",
    [
        (np.zeros(5), 2, "Input axis should be 0 or 1"),
        ([0.0, 1.0], 0, "Unsupported inner type"),
    ],
    ids=["axis", "type"],
)
def test_convert_X_guards(X_in, axis, message):
    with pytest.raises(ValueError, match=message):
        _Segmenter()._convert_X(X_in, axis)


# ---------------------------------------------------------------------------
# fit / predict flow
# ---------------------------------------------------------------------------


def test_fit_validates_the_parameter_schema(X):
    with pytest.raises(ValueError, match="parameter validation failed"):
        _Segmenter(window=0).fit(X)
    with pytest.raises(ValueError, match="window must fit in the series"):
        _Segmenter(window=N_TIMEPOINTS).fit(X)


def test_fit_predict_and_predict_on_a_new_series(X):
    est = _Segmenter()
    np.testing.assert_array_equal(est.fit_predict(X), [N_TIMEPOINTS // 2])
    np.testing.assert_array_equal(est.predict(X[:10]), [5])
    assert est.metadata_["n_channels"] == 3
    assert est._data_context["n_samples"] == 10  # refreshed by predict


def test_fit_predict_passes_the_axis_on(X):
    est = _Segmenter()
    np.testing.assert_array_equal(est.fit_predict(X.T, axis=1), [N_TIMEPOINTS // 2])
    np.testing.assert_array_equal(est.X_predict_, X)


def test_empty_fit_checks_nothing_until_predict(X):
    est = _EmptyFitSegmenter(window=0)
    assert est.fit("not a series") is est
    assert est.is_fitted
    assert not hasattr(est, "X_fit_")
    with pytest.raises(ValueError, match="parameter validation failed"):
        est.predict(X)


def test_empty_fit_predict_does_not_require_fit(X):
    est = _EmptyFitSegmenter()
    np.testing.assert_array_equal(est.predict(X), [N_TIMEPOINTS // 2])
    assert not est.is_fitted


def test_requires_y_tag(X):
    with pytest.raises(ValueError, match="requires_y is true"):
        _RequiresYSegmenter().fit(X)
    assert _RequiresYSegmenter().fit(X, y=np.zeros(N_TIMEPOINTS)).is_fitted


def test_default_fit_hook(X):
    x = X[:, 0]
    est = _DefaultFitSegmenter()
    assert est._fit(x, None) is est
    assert est.fit(x).is_fitted
    assert est.predict(x).size == 0


def test_get_parameter_schema_is_a_merged_copy():
    schema = _UnivariateSegmenter.get_parameter_schema()
    assert set(schema) == {"window", "_cross_constraints"}
    schema["_cross_constraints"].clear()
    assert len(_Segmenter.get_parameter_schema()["_cross_constraints"]) == 1


# ---------------------------------------------------------------------------
# Label converters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "change_points, expected",
    [([2, 5], [0, 0, 1, 0, 0, 1, 0]), ([], [0] * 7)],
    ids=["two", "none"],
)
def test_to_classification(change_points, expected):
    labels = BaseSegmenter.to_classification(change_points, 7)
    assert labels.dtype.kind == "i"
    np.testing.assert_array_equal(labels, expected)


@pytest.mark.parametrize(
    "change_points, expected",
    [([2, 5], [0, 0, 1, 1, 1, 2, 2]), ([], [0] * 7)],
    ids=["two", "none"],
)
def test_to_clusters(change_points, expected):
    labels = BaseSegmenter.to_clusters(change_points, 7)
    assert labels.dtype.kind == "i"
    np.testing.assert_array_equal(labels, expected)
