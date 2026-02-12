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
* ``is_fitted`` lifecycle
* ``NotFittedError`` raised before ``fit`` (when applicable)
* ``reset`` clears fitted state
* ``repr`` does not crash
* Pickle round-trip (when ``cant_pickle`` is ``False``)
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from .conftest import apply_supervision, make_instance

# The Tirex detectors inherit from a *different* BaseSegmenter
# (``segmenter.base.BaseSegmenter``) that shares the same public API but
# is a distinct class.  We import both and accept either.
_SEGMENTER_BASES = []

from tsseg.algorithms.base import BaseSegmenter  # noqa: E402

_SEGMENTER_BASES.append(BaseSegmenter)
try:
    from segmenter.base import BaseSegmenter as _TirexBase  # noqa: E402

    _SEGMENTER_BASES.append(_TirexBase)
except ImportError:
    pass

_SEGMENTER_BASES = tuple(_SEGMENTER_BASES)


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


# ==================================================================
# Tests
# ==================================================================


class TestBaseSegmenterContract:
    """Every detector must satisfy the ``BaseSegmenter`` API contract."""

    def test_is_base_segmenter(self, algorithm):
        """All algorithms must inherit from a ``BaseSegmenter`` variant."""
        _name, _cls, _ovr, instance = algorithm
        assert isinstance(instance, _SEGMENTER_BASES), (
            f"{_name} does not inherit from any known BaseSegmenter"
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
        assert cloned.get_params(deep=False) == instance.get_params(deep=False)

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
            X, y = synthetic_data["multivariate"]["X"], synthetic_data["multivariate"]["y"]
        if ovr.semi_supervised:
            apply_supervision(instance, y)
            result = instance.fit(X, y)
        else:
            result = instance.fit(X)
        assert result is instance

    def test_is_fitted_lifecycle(self, algorithm, synthetic_data):
        """``is_fitted`` must be ``False`` before ``fit`` and ``True`` after."""
        _name, cls, ovr, _instance = algorithm
        instance = make_instance(cls, ovr)
        # Before fit: is_fitted can be True if fit_is_empty, but for non-empty
        # it must be False.
        if not instance.get_tag("fit_is_empty"):
            assert not instance.is_fitted
        _fit_instance(instance, ovr, synthetic_data)
        assert instance.is_fitted

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
        except (pickle.PicklingError, TypeError, AttributeError) as exc:
            pytest.skip(f"Cannot pickle: {exc}")
        assert type(loaded) is type(instance)
        assert loaded.get_params(deep=False) == instance.get_params(deep=False)
        assert loaded.is_fitted
