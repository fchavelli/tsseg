"""Input validation and robustness tests.

Verify that detectors raise clear errors on invalid inputs and correctly
reject data that doesn't match their declared capabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from .conftest import make_instance


class TestInputValidation:
    """Detectors must reject bad inputs with meaningful errors."""

    def test_invalid_input_type_raises(self, algorithm):
        """Passing a plain list (not array) should raise ``ValueError``."""
        _name, _cls, _ovr, instance = algorithm
        with pytest.raises((ValueError, TypeError)):
            instance.fit_predict([[1, 2], [3, 4]])

    def test_string_input_raises(self, algorithm):
        """String arrays must be rejected."""
        _name, _cls, _ovr, instance = algorithm
        X = np.array(["a", "b", "c"])
        with pytest.raises((ValueError, TypeError)):
            instance.fit_predict(X)

    def test_empty_input_raises(self, algorithm):
        """Empty arrays should raise or return empty — not crash silently."""
        _name, _cls, _ovr, instance = algorithm
        X = np.empty((0, 1))
        try:
            result = instance.fit_predict(X)
            # If it doesn't raise, the result must be empty too
            assert np.asarray(result).size == 0
        except (ValueError, IndexError):
            pass  # Acceptable: raising on empty input

    def test_multivariate_rejected_when_unsupported(self, algorithm):
        """Multivariate input must raise when ``capability:multivariate``
        is ``False``."""
        _name, cls, ovr, instance = algorithm
        if instance.get_tag("capability:multivariate"):
            pytest.skip("Algorithm supports multivariate")
        X = np.random.default_rng(0).standard_normal((100, 3))
        instance = make_instance(cls, ovr)
        with pytest.raises(ValueError, match="(?i)multivariate"):
            instance.fit_predict(X)

    def test_nan_rejected_when_unsupported(self, algorithm, synthetic_data):
        """NaN values must raise when ``capability:missing_values`` is ``False``."""
        _name, cls, ovr, instance = algorithm
        if instance.get_tag("capability:missing_values"):
            pytest.skip("Algorithm supports missing values")
        if instance.get_tag("capability:univariate"):
            X = synthetic_data["univariate"]["X"].copy()
        else:
            X = synthetic_data["multivariate"]["X"].copy()
        X[50] = np.nan
        instance = make_instance(cls, ovr)
        with pytest.raises(ValueError, match="(?i)missing"):
            instance.fit_predict(X)

    def test_pandas_series_accepted(self, algorithm):
        """A ``pd.Series`` is a valid input type."""
        _name, cls, ovr, instance = algorithm
        if not instance.get_tag("capability:univariate"):
            pytest.skip("Multivariate-only")
        instance = make_instance(cls, ovr)
        X = pd.Series(np.random.default_rng(0).standard_normal(200))
        try:
            result = instance.fit_predict(X)
            assert result is not None
        except NotImplementedError:
            pytest.skip("Algorithm does not support this input pathway")

    def test_pandas_dataframe_accepted(self, algorithm):
        """A ``pd.DataFrame`` is a valid input type."""
        _name, cls, ovr, instance = algorithm
        if not instance.get_tag("capability:univariate"):
            pytest.skip("Multivariate-only")
        instance = make_instance(cls, ovr)
        X = pd.DataFrame({"x": np.random.default_rng(0).standard_normal(200)})
        try:
            result = instance.fit_predict(X)
            assert result is not None
        except NotImplementedError:
            pytest.skip("Algorithm does not support this input pathway")
