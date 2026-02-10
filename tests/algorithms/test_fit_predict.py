import numpy as np
import pytest


def _get_config_value(config, name, default=None):
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, dict):
        return config.get(name, default)
    return default


def _get_dataset_or_skip(fit_predict_result, key: str) -> dict:
    datasets = fit_predict_result["results"]
    if key not in datasets:
        pytest.skip(f"{key.capitalize()} data not supported by this algorithm")
    return datasets[key]


def _assert_output_exists(result):
    assert result is not None, "fit_predict returned None"

    if isinstance(result, np.ndarray):
        assert result.size > 0 or result.shape == (0,), "Empty output must be an array"
    elif isinstance(result, list):
        assert len(result) > 0 or result == [], "Empty output must be a list"
    else:
        array_like = np.asarray(result)
        assert array_like.size > 0, "fit_predict returned empty output"


def _assert_output_shape(instance, dataset):
    returns_dense = instance.get_tag("returns_dense")
    result = np.asarray(dataset["result"])
    n_samples = dataset["X"].shape[0]

    if returns_dense:
        assert result.ndim == 1, "Expected 1D array of change points"
        if result.size > 0:
            assert np.all(result >= 0), "Change point indices must be non-negative"
            assert np.all(result <= n_samples), "Change point indices must not exceed series length"
    else:
        assert result.shape[0] == n_samples, (
            f"State labels length ({result.shape[0]}) must match "
            f"number of samples ({n_samples})"
        )
def _assert_detection_quality(instance, config, dataset):
    if _get_config_value(config, "skip_detection", False):
        pytest.skip("Detection accuracy skipped for this algorithm")

    returns_dense = instance.get_tag("returns_dense")
    result = np.asarray(dataset["result"])
    tolerance = _get_config_value(config, "detection_tolerance", 0)
    change_points = dataset["change_points"]
    n_samples = dataset["X"].shape[0]

    if returns_dense:
        if result.size == 0:
            pytest.fail("No change points detected")
        detected = np.unique(result)
        if detected.size == 0:
            pytest.fail("No change points detected")
    else:
        diffs = np.diff(result, axis=0)
        if diffs.ndim == 1:
            change_mask = diffs != 0
        else:
            change_mask = np.any(diffs != 0, axis=tuple(range(1, diffs.ndim)))
        detected = np.flatnonzero(change_mask) + 1

    if detected.size == 0:
        flat_states = result.reshape(result.shape[0], -1)
        unique_states = np.unique(flat_states, axis=0)
        if unique_states.shape[0] < 2:
            pytest.fail("No state transitions or label diversity detected")
        return

    has_expected_match = False
    for expected in change_points:
        if expected >= n_samples:
            continue
        if np.any(np.abs(detected - expected) <= tolerance):
            has_expected_match = True
            break

    if not has_expected_match and not returns_dense:
        flat_states = result.reshape(result.shape[0], -1)
        unique_states = np.unique(flat_states, axis=0)
        if unique_states.shape[0] < 2:
            pytest.fail(
                "Detected transitions do not align with expected change points and only one state was predicted"
            )


def test_univariate_output_exists(fit_predict_result):
    dataset = _get_dataset_or_skip(fit_predict_result, "univariate")
    _assert_output_exists(dataset["result"])


def test_univariate_output_shape(fit_predict_result):
    instance = fit_predict_result["instance"]
    dataset = _get_dataset_or_skip(fit_predict_result, "univariate")
    _assert_output_shape(instance, dataset)


def test_univariate_detects_change_points(fit_predict_result):
    instance = fit_predict_result["instance"]
    config = fit_predict_result["config"]
    dataset = _get_dataset_or_skip(fit_predict_result, "univariate")
    _assert_detection_quality(instance, config, dataset)


def test_multivariate_output_exists(fit_predict_result):
    dataset = _get_dataset_or_skip(fit_predict_result, "multivariate")
    _assert_output_exists(dataset["result"])


def test_multivariate_output_shape(fit_predict_result):
    instance = fit_predict_result["instance"]
    dataset = _get_dataset_or_skip(fit_predict_result, "multivariate")
    _assert_output_shape(instance, dataset)


def test_multivariate_detects_change_points(fit_predict_result):
    instance = fit_predict_result["instance"]
    config = fit_predict_result["config"]
    dataset = _get_dataset_or_skip(fit_predict_result, "multivariate")
    _assert_detection_quality(instance, config, dataset)
