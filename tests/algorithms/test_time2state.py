"""Behaviour of Time2StateDetector not covered by the generic suite."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsseg.algorithms import Time2StateDetector  # noqa: E402

WINDOW_SIZE = 32
SMALL = {
    "window_size": WINDOW_SIZE,
    "n_states": 2,
    "nb_steps": 2,
    "random_state": 0,
}


class _SplitClustering:
    """Stand-in for the DPGMM: the second half of the windows is state 1."""

    def fit(self, X):
        n_windows = len(X)
        return (np.arange(n_windows) >= n_windows // 2).astype(int)


def _two_regime_series(n_timepoints, rng=0):
    rng = np.random.default_rng(rng)
    half = n_timepoints // 2
    return np.concatenate(
        [rng.normal(0.0, 0.3, half), rng.normal(2.0, 0.3, n_timepoints - half)]
    )


@pytest.mark.parametrize(
    ("n_timepoints", "step"),
    [(300, 20), (292, 20), (300, 30), (301, 3)],
    ids=["tail-8", "no-tail", "tail-28", "tail-2"],
)
def test_points_after_the_last_window_take_its_label(n_timepoints, step):
    X = _two_regime_series(n_timepoints)
    detector = Time2StateDetector(step=step, **SMALL).fit(X)
    detector.clustering_ = _SplitClustering()

    labels = detector.predict(X)

    n_windows = (n_timepoints - WINDOW_SIZE) // step + 1
    covered = (n_windows - 1) * step + WINDOW_SIZE
    assert covered == n_timepoints - (n_timepoints - WINDOW_SIZE) % step
    assert labels.shape == (n_timepoints,)
    assert labels[0] == 0
    # The last window, labelled 1, alone covers the last covered point.
    assert labels[covered - 1] == 1
    # The points that no window covers extend the vote of the last window.
    np.testing.assert_array_equal(labels[covered:], 1)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"window_size": 33}, "window_size must be even"),
        ({"kernel_size": 1}, "kernel_size"),
    ],
    ids=["odd-window", "kernel-1"],
)
def test_fit_refuses_parameters_the_encoder_cannot_run(params, message):
    # An odd window used to pass fit and fail in predict, after the training;
    # kernel_size=1 used to fail in fit with a RuntimeError from torch.
    detector = Time2StateDetector(**{**SMALL, "step": 16, **params})

    with pytest.raises(ValueError, match=message):
        detector.fit(_two_regime_series(300))


@pytest.mark.parametrize(
    "params",
    [{"window_size": 4, "step": 2}, {"kernel_size": 2}, {"kernel_size": 64}],
    ids=["window-4", "kernel-2", "kernel-larger-than-window"],
)
def test_schema_accepts_parameters_the_encoder_can_run(params):
    from tsseg.algorithms.param_schema import validate_params

    detector = Time2StateDetector(**{**SMALL, "step": 16, **params})

    assert validate_params(detector, {"n_samples": 300, "n_channels": 1}) == []


@pytest.mark.parametrize(
    "value", [0.0, 0.1], ids=["zero-variance", "rounding-variance"]
)
def test_a_constant_series_is_one_state(value):
    # np.full(300, 0.0) has a variance of exactly 0: the z-normalisation
    # divided by zero and the DPGMM refused the NaN. np.full(300, 0.1) has a
    # variance of a few ulps and was already labelled as one state.
    X = np.full(300, value)

    labels = Time2StateDetector(**{**SMALL, "step": 16}).fit_predict(X)

    np.testing.assert_array_equal(labels, np.zeros(300, dtype=int))


def test_predict_labels_a_constant_series_as_one_state():
    detector = Time2StateDetector(**{**SMALL, "step": 16}).fit(_two_regime_series(300))

    labels = detector.predict(np.zeros(300))

    np.testing.assert_array_equal(labels, np.zeros(300, dtype=int))
