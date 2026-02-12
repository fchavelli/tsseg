"""Prediction output contract tests.

This module verifies that ``fit_predict`` and ``fit`` + ``predict`` produce
outputs that conform to the detector's declared contract (``returns_dense``
tag, output shape, value range, dtype).

Each test is parametrised over every algorithm via the ``algorithm`` fixture.
Both univariate and multivariate inputs are exercised when supported.
"""

from __future__ import annotations

import numpy as np
import pytest

from .conftest import apply_supervision, make_instance


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_fit_predict(cls, ovr, data_dict):
    """Return ``(instance, result)`` after running ``fit_predict``."""
    instance = make_instance(cls, ovr)
    X, y = data_dict["X"], data_dict["y"]
    if ovr.semi_supervised:
        apply_supervision(instance, y)
        result = instance.fit_predict(X, y)
    else:
        result = instance.fit_predict(X)
    return instance, result


def _run_fit_then_predict(cls, ovr, data_dict):
    """Return ``(instance, result)`` via ``fit`` then ``predict``."""
    instance = make_instance(cls, ovr)
    X, y = data_dict["X"], data_dict["y"]
    if ovr.semi_supervised:
        apply_supervision(instance, y)
        instance.fit(X, y)
    else:
        instance.fit(X)
    result = instance.predict(X)
    return instance, result


def _pick_data(instance, ovr, synthetic_data):
    """Return the first suitable data dict for the algorithm."""
    if instance.get_tag("capability:univariate"):
        return synthetic_data["univariate"]
    return synthetic_data["multivariate"]


# ==================================================================
# Output shape / type / value-range
# ==================================================================


class TestPredictOutputContract:
    """Verify that predictions satisfy the declared output contract."""

    def test_output_is_array_like(self, algorithm, synthetic_data):
        """Prediction must be convertible to a numpy array."""
        _name, cls, ovr, instance = algorithm
        data = _pick_data(instance, ovr, synthetic_data)
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        assert arr.ndim >= 1, "Output must have at least one dimension"

    def test_dense_change_points_shape(self, algorithm, synthetic_data):
        """If ``returns_dense=True``: 1-D array of indices in ``[0, N)``."""
        _name, cls, ovr, instance = algorithm
        if not instance.get_tag("returns_dense"):
            pytest.skip("Not a sparse change-point detector")
        data = _pick_data(instance, ovr, synthetic_data)
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        n = data["X"].shape[0]
        assert arr.ndim == 1, f"Expected 1-D output, got shape {arr.shape}"
        if arr.size > 0:
            assert np.all(arr >= 0), "Change-point indices must be ≥ 0"
            assert np.all(arr < n), f"Change-point indices must be < {n}"

    def test_state_labels_shape(self, algorithm, synthetic_data):
        """If ``returns_dense=False``: array of length ``N``."""
        _name, cls, ovr, instance = algorithm
        if instance.get_tag("returns_dense"):
            pytest.skip("Not a state-label detector")
        data = _pick_data(instance, ovr, synthetic_data)
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        n = data["X"].shape[0]
        assert arr.shape[0] == n, (
            f"State labels length ({arr.shape[0]}) != number of samples ({n})"
        )

    def test_change_points_are_sorted(self, algorithm, synthetic_data):
        """If ``returns_dense=True``: indices must be sorted."""
        _name, cls, ovr, instance = algorithm
        if not instance.get_tag("returns_dense"):
            pytest.skip("Not a sparse change-point detector")
        data = _pick_data(instance, ovr, synthetic_data)
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        if arr.size > 1:
            assert np.all(np.diff(arr) >= 0), "Change-point indices must be sorted"

    def test_state_labels_dtype(self, algorithm, synthetic_data):
        """If ``returns_dense=False``: labels should be integer-like."""
        _name, cls, ovr, instance = algorithm
        if instance.get_tag("returns_dense"):
            pytest.skip("Not a state-label detector")
        data = _pick_data(instance, ovr, synthetic_data)
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        assert np.issubdtype(arr.dtype, np.integer) or np.allclose(
            arr, arr.astype(int)
        ), "State labels must be integer-valued"


# ==================================================================
# Consistency: fit_predict == fit + predict
# ==================================================================


class TestFitPredictConsistency:
    """``fit_predict(X)`` must return the same result as ``fit(X); predict(X)``
    for deterministic algorithms."""

    def test_fit_predict_equals_fit_then_predict(self, algorithm, synthetic_data):
        _name, cls, ovr, instance = algorithm
        if instance.get_tag(
            "non_deterministic", raise_error=False, tag_value_default=False
        ):
            pytest.skip("Non-deterministic algorithm")
        data = _pick_data(instance, ovr, synthetic_data)

        _, result_fp = _run_fit_predict(cls, ovr, data)
        _, result_sep = _run_fit_then_predict(cls, ovr, data)
        np.testing.assert_array_equal(
            np.asarray(result_fp),
            np.asarray(result_sep),
            err_msg="fit_predict != fit + predict",
        )


# ==================================================================
# Determinism: same input → same output
# ==================================================================


class TestDeterminism:
    """Deterministic algorithms must produce identical outputs on repeated
    calls with the same input."""

    def test_deterministic_output(self, algorithm, synthetic_data):
        _name, cls, ovr, instance = algorithm
        if instance.get_tag(
            "non_deterministic", raise_error=False, tag_value_default=False
        ):
            pytest.skip("Non-deterministic algorithm")
        data = _pick_data(instance, ovr, synthetic_data)

        _, r1 = _run_fit_predict(cls, ovr, data)
        _, r2 = _run_fit_predict(cls, ovr, data)
        np.testing.assert_array_equal(
            np.asarray(r1), np.asarray(r2), err_msg="Results differ across runs"
        )


# ==================================================================
# Multivariate support
# ==================================================================


class TestMultivariateSupport:
    """Algorithms declaring ``capability:multivariate`` must handle it."""

    def test_multivariate_runs(self, algorithm, synthetic_data):
        _name, cls, ovr, instance = algorithm
        if not instance.get_tag("capability:multivariate"):
            pytest.skip("Univariate-only algorithm")
        data = synthetic_data["multivariate"]
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        assert arr.size > 0, "Multivariate prediction returned empty output"

    def test_univariate_runs(self, algorithm, synthetic_data):
        _name, cls, ovr, instance = algorithm
        if not instance.get_tag("capability:univariate"):
            pytest.skip("Multivariate-only algorithm")
        data = synthetic_data["univariate"]
        _, result = _run_fit_predict(cls, ovr, data)
        arr = np.asarray(result)
        assert arr.size > 0, "Univariate prediction returned empty output"
