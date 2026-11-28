"""Independent window, Gaussian and label-placement oracles for TICC."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import multivariate_normal

from tsseg.algorithms import TiccDetector
from tsseg.algorithms.ticc.ticc import TICC, getTrainTestSplit


@pytest.mark.parametrize("window_size", [1, 3, 5])
@pytest.mark.parametrize("n_channels", [1, 2])
def test_training_rows_are_complete_windows(window_size, n_channels):
    X = np.arange(1.0, 9 * n_channels + 1).reshape(9, n_channels)
    starts = getTrainTestSplit(
        len(X), window_size + 1, window_size, np.random.RandomState(102)
    )
    solver = TICC(window_size=window_size)

    actual = solver.stack_training_data(X, n_channels, len(starts), starts)
    expected = np.array([X[t : t + window_size].ravel() for t in starts])

    assert_array_equal(actual, expected)
    assert_array_equal(actual[-1, -n_channels:], X[-1])


def test_window_starts_need_not_be_consecutive():
    X = np.arange(1.0, 9.0)[:, None]
    starts = [0, 3, 5]
    actual = TICC(window_size=3).stack_training_data(X, 1, len(starts), starts)

    assert_array_equal(actual, [[1, 2, 3], [4, 5, 6], [6, 7, 8]])


def _known_model(window_size):
    """Known diagonal Gaussian states; state 1 occurs only at the end."""
    n_channels, n_windows = 2, 7
    dim = window_size * n_channels
    D = np.arange(1.0, n_windows * dim + 1).reshape(n_windows, dim) / dim
    covariances = {0: np.diag(np.linspace(0.5, 1.5, dim)), 1: np.eye(dim) * 2}
    means = {0: np.zeros(dim), 1: np.full(dim, 5.0)}
    labels = np.array([0, 0, 0, 0, 0, 0, 1])
    solver = TICC(window_size=window_size, number_of_clusters=2, beta=0)
    solver.trained_model = {
        "cluster_mean_info": {},
        "computed_covariance": {(2, c): covariances[c] for c in (0, 1)},
        "cluster_mean_stacked_info": {(2, c): means[c] for c in (0, 1)},
        "complete_D_train": D,
        "time_series_col_size": n_channels,
    }
    solver.train_cluster_inverse_ = {c: np.linalg.inv(covariances[c]) for c in (0, 1)}
    solver.clustered_points_ = labels
    logpdf = np.column_stack(
        [
            multivariate_normal.logpdf(D, mean=means[c], cov=covariances[c])
            for c in (0, 1)
        ]
    )
    return solver, logpdf


@pytest.mark.parametrize("window_size", [1, 3, 5])
def test_every_window_has_its_gaussian_assignment_cost(window_size):
    solver, logpdf = _known_model(window_size)
    model = solver.trained_model
    actual = solver.smoothen_clusters(
        model["cluster_mean_info"],
        model["computed_covariance"],
        model["cluster_mean_stacked_info"],
        model["complete_D_train"],
        model["time_series_col_size"],
    )
    dim = model["complete_D_train"].shape[1]

    assert_allclose(actual, -2 * logpdf - dim * np.log(2 * np.pi), atol=1e-12)


@pytest.mark.parametrize("window_size", [1, 3, 5])
def test_log_likelihood_includes_every_complete_window(window_size):
    solver, logpdf = _known_model(window_size)
    labels = solver.clustered_points_
    expected = logpdf[np.arange(len(labels)), labels].sum()

    assert_allclose(solver.log_likelihood(), expected, atol=1e-12)


def test_beta_zero_assigns_tail_windows_to_their_most_likely_state():
    solver, logpdf = _known_model(3)
    expected = logpdf.argmax(axis=1)
    assert_array_equal(expected[-2:], [1, 1])

    assert_array_equal(solver.predict_clusters(), expected)


def test_bic_counts_a_state_that_occurs_only_in_the_last_windows():
    labels = np.array([0, 0, 0, 0, 1, 1])
    model = SimpleNamespace(
        train_cluster_inverse_={0: np.eye(3), 1: np.eye(3) * 2},
        log_likelihood=lambda: -100.0,
    )
    detector = TiccDetector(window_size=3, n_states=2)
    # Each state: one Toeplitz diagonal parameter and one channel mean;
    # add one switch. The criterion keeps its existing composite scaling.
    expected = 200 / 3 + (2 * 2 + 1) * np.log(8)

    assert_allclose(detector._bic(model, labels, 8, 1), expected)


@pytest.mark.parametrize("window_size", [1, 3, 5])
def test_point_labels_keep_start_alignment_and_extend_the_final_label(
    monkeypatch, window_size
):
    X = np.arange(1.0, 10.0)[:, None]
    n_windows = len(X) - window_size + 1
    window_labels = np.arange(n_windows) % 2
    monkeypatch.setattr(
        TiccDetector,
        "_fit_ticc",
        lambda self, series, n_states: (object(), window_labels.copy()),
    )

    actual = TiccDetector(window_size=window_size, n_states=2).fit_predict(X)

    assert_array_equal(actual[:n_windows], window_labels)
    assert_array_equal(
        actual[n_windows:], np.repeat(window_labels[-1], window_size - 1)
    )
