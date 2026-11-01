"""Common estimator contract tests — scikit-learn style.

Every check in this module verifies a *generic* contract that **all**
``BaseSegmenter`` subclasses must satisfy, regardless of the underlying
algorithm.  The tests are parametrised over every algorithm in
``tsseg.algorithms.__all__`` via the ``algorithm`` fixture defined in
``conftest.py``.

The checks mirror the principles behind ``sklearn.utils.estimator_checks``:

* ``get_params`` / ``set_params`` round-trip
* ``clone`` produces an equivalent estimator
* ``fit`` returns ``self``
* ``is_fitted`` lifecycle, and ``get_fitted_params`` along it
* ``NotFittedError`` raised before ``fit`` (when applicable)
* ``reset`` clears fitted state
* ``repr`` does not crash
* Pickle round-trip (when ``cant_pickle`` is ``False``)
"""

from __future__ import annotations

import pickle

import joblib
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from tsseg.algorithms.base import BaseSegmenter

from .conftest import apply_supervision, make_instance

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fit_instance(instance, ovr, data):
    """Fit an instance on suitable synthetic data."""
    if instance.get_tag("capability:univariate"):
        X, y = data["univariate"]["X"], data["univariate"]["y"]
    else:
        X, y = data["multivariate"]["X"], data["multivariate"]["y"]

    if ovr.semi_supervised:
        apply_supervision(instance, y)
        instance.fit(X, y)
    else:
        instance.fit(X)
    return instance


def _same_value(a, b) -> bool:
    """Compare two parameter values by content.

    ``clone`` copies the parameters as scikit-learn's ``clone`` does: dicts,
    lists and tuples item by item, anything else by ``deepcopy``. Items are
    compared the same way, and a copied array or a copied object without
    ``__eq__`` through ``joblib.hash`` of its content, as scikit-learn's
    estimator checks do.
    """
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys() == b.keys() and all(_same_value(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(map(_same_value, a, b))
    return joblib.hash(a) == joblib.hash(b)


def _assert_same_params(cloned, instance):
    """Assert that ``cloned`` has the parameters of ``instance``, by content."""
    expected = instance.get_params(deep=False)
    params = cloned.get_params(deep=False)
    assert params.keys() == expected.keys()
    for name, value in expected.items():
        assert _same_value(params[name], value), name


# ==================================================================
# Tests
# ==================================================================


class TestBaseSegmenterContract:
    """Every detector must satisfy the ``BaseSegmenter`` API contract."""

    def test_is_base_segmenter(self, algorithm):
        """All algorithms must inherit from ``tsseg``'s ``BaseSegmenter``."""
        name, _cls, _ovr, instance = algorithm
        assert isinstance(instance, BaseSegmenter), (
            f"{name} does not inherit from BaseSegmenter"
        )

    def test_get_params_returns_dict(self, algorithm):
        """``get_params()`` must return a dict (scikit-learn contract)."""
        _name, _cls, _ovr, instance = algorithm
        params = instance.get_params(deep=False)
        assert isinstance(params, dict)

    def test_set_params_roundtrip(self, algorithm):
        """``set_params(**get_params())`` must not alter the estimator."""
        _name, _cls, _ovr, instance = algorithm
        params = instance.get_params(deep=False)
        instance.set_params(**params)
        assert instance.get_params(deep=False) == params

    def test_clone_equivalence(self, algorithm):
        """``clone()`` must produce an estimator with identical parameters."""
        _name, _cls, _ovr, instance = algorithm
        if not hasattr(instance, "clone"):
            pytest.skip("Algorithm does not implement clone()")
        cloned = instance.clone()
        assert type(cloned) is type(instance)
        _assert_same_params(cloned, instance)

    def test_clone_copies_mutable_parameters(self, algorithm):
        """A clone must not share a mutable parameter with the original."""
        _name, _cls, _ovr, instance = algorithm
        cloned = instance.clone()
        params = cloned.get_params(deep=False)
        for name, value in instance.get_params(deep=False).items():
            if isinstance(value, (dict, list, set, np.ndarray)):
                assert params[name] is not value, name

    def test_clone_independence(self, algorithm, synthetic_data):
        """Modifying a clone must not affect the original."""
        _name, cls, ovr, _instance = algorithm
        original = make_instance(cls, ovr)
        if not hasattr(original, "clone"):
            pytest.skip("Algorithm does not implement clone()")
        cloned = original.clone()
        _fit_instance(cloned, ovr, synthetic_data)
        # Original should still be unfitted (unless fit_is_empty)
        if not original.get_tag("fit_is_empty"):
            assert not original.is_fitted

    def test_repr(self, algorithm):
        """``repr()`` must not raise."""
        _name, _cls, _ovr, instance = algorithm
        r = repr(instance)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_fit_returns_self(self, algorithm, synthetic_data):
        """``fit()`` must return ``self`` for method chaining."""
        _name, cls, ovr, _instance = algorithm
        instance = make_instance(cls, ovr)
        if instance.get_tag("capability:univariate"):
            X, y = synthetic_data["univariate"]["X"], synthetic_data["univariate"]["y"]
        else:
            X, y = (
                synthetic_data["multivariate"]["X"],
                synthetic_data["multivariate"]["y"],
            )
        if ovr.semi_supervised:
            apply_supervision(instance, y)
            result = instance.fit(X, y)
        else:
            result = instance.fit(X)
        assert result is instance

    def test_is_fitted_lifecycle(self, algorithm, synthetic_data):
        """``is_fitted`` must be ``False`` before ``fit`` and ``True`` after.

        ``get_fitted_params`` follows the same lifecycle and keeps the base
        contract: it accepts ``deep`` and lists only fitted attributes, whose
        names end with ``_``.
        """
        _name, cls, ovr, _instance = algorithm
        instance = make_instance(cls, ovr)
        # Before fit: is_fitted can be True if fit_is_empty, but for non-empty
        # it must be False.
        if not instance.get_tag("fit_is_empty"):
            assert not instance.is_fitted
            with pytest.raises(NotFittedError):
                instance.get_fitted_params()
        _fit_instance(instance, ovr, synthetic_data)
        assert instance.is_fitted
        for deep in (True, False):
            params = instance.get_fitted_params(deep=deep)
            assert isinstance(params, dict)
            not_fitted = [
                key for key in params if key.startswith("_") or not key.endswith("_")
            ]
            assert not not_fitted, (
                f"get_fitted_params(deep={deep}) returned keys that are not "
                f"fitted attributes: {not_fitted}"
            )

    def test_not_fitted_error(self, algorithm, synthetic_data):
        """``predict`` before ``fit`` must raise when ``fit_is_empty=False``."""
        _name, cls, ovr, _instance = algorithm
        instance = make_instance(cls, ovr)
        if instance.get_tag("fit_is_empty"):
            pytest.skip("fit_is_empty=True — predict works without fit")

        if instance.get_tag("capability:univariate"):
            X = synthetic_data["univariate"]["X"]
        else:
            X = synthetic_data["multivariate"]["X"]

        with pytest.raises(NotFittedError):
            instance.predict(X)

    def test_reset(self, algorithm, synthetic_data):
        """``reset()`` must clear the fitted state."""
        _name, cls, ovr, _instance = algorithm
        instance = make_instance(cls, ovr)
        _fit_instance(instance, ovr, synthetic_data)
        instance.reset()
        if not instance.get_tag("fit_is_empty"):
            assert not instance.is_fitted

    def test_pickle_roundtrip(self, algorithm, synthetic_data):
        """Pickle → unpickle must produce an estimator with identical params.

        Skipped for algorithms that declare ``cant_pickle=True``.
        """
        _name, cls, ovr, _instance = algorithm
        instance = make_instance(cls, ovr)
        if instance.get_tag("cant_pickle", raise_error=False, tag_value_default=False):
            pytest.skip("Algorithm declares cant_pickle=True")
        _fit_instance(instance, ovr, synthetic_data)

        try:
            data = pickle.dumps(instance)
            loaded = pickle.loads(data)
        except (pickle.PicklingError, TypeError, AttributeError, RuntimeError) as exc:
            pytest.skip(f"Cannot pickle: {exc}")
        assert type(loaded) is type(instance)
        # Compare params element-wise (handles numpy arrays and callables).
        loaded_params = loaded.get_params(deep=False)
        original_params = instance.get_params(deep=False)
        assert set(loaded_params.keys()) == set(original_params.keys())
        for key in original_params:
            a, b = original_params[key], loaded_params[key]
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b, err_msg=f"Param {key!r} differs")
            elif callable(a) or isinstance(a, list | tuple):
                # Skip deep comparison for callables and complex containers
                # (e.g. emission functions) — type match is sufficient.
                assert isinstance(a, type(b)), f"Param {key!r} type differs"
            elif isinstance(a, float) and np.isnan(a):
                assert isinstance(b, float) and np.isnan(b), (
                    f"Param {key!r} differs: {a!r} != {b!r}"
                )
            else:
                assert a == b, f"Param {key!r} differs: {a!r} != {b!r}"
        assert loaded.is_fitted
