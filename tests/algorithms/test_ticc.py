"""Tests of TiccDetector: solver output, number of states chosen by BIC."""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import TiccDetector
from tsseg.algorithms.ticc.ticc import n_free_parameters


def _series():
    """Three regimes of 200 points: x2 follows x1, then -x1, then x1."""
    rng = np.random.default_rng(0)
    parts = []
    for sign in (1, -1, 1):
        x1 = rng.normal(0, 1, 200)
        x2 = sign * x1 + rng.normal(0, 0.3, 200)
        parts.append(np.column_stack([x1, x2]))
    return np.concatenate(parts)


def _runs(labels):
    """Change points and the label of each segment."""
    cps = np.flatnonzero(np.diff(labels)) + 1
    return cps.tolist(), [int(labels[0])] + [int(labels[c]) for c in cps]


# Outputs of the solver before its refactoring (tsseg 0.3.0) on _series().
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(window_size=3, n_states=2, maxIters=20),
            ([199, 399], [0, 1, 0]),
        ),
        (
            dict(
                window_size=5, n_states=3, lambda_parameter=1e-2, beta=100, maxIters=20
            ),
            ([198, 266, 286, 399], [2, 1, 0, 1, 2]),
        ),
    ],
)
def test_guided_output_is_unchanged(kwargs, expected):
    labels = TiccDetector(**kwargs).fit_predict(_series())

    assert _runs(labels) == expected


def test_bic_selects_the_number_of_states():
    detector = TiccDetector(window_size=3, maxIters=20, max_states=4)

    labels = detector.fit_predict(_series())

    assert detector.n_states_ == 2
    assert sorted(detector.bic_) == [1, 2, 3, 4]
    assert min(detector.bic_, key=detector.bic_.get) == 2
    assert _runs(labels) == ([199, 399], [0, 1, 0])


def test_guided_mode_has_no_bic():
    detector = TiccDetector(window_size=3, n_states=2, maxIters=5)

    detector.fit(_series())

    assert detector.n_states_ == 2
    assert not hasattr(detector, "bic_")


def test_warns_when_bic_selects_max_states():
    detector = TiccDetector(window_size=3, maxIters=20, max_states=2)

    with pytest.warns(UserWarning, match="max_states"):
        detector.fit(_series())

    assert detector.n_states_ == 2


def test_leaves_numpy_global_state_untouched():
    np.random.seed(7)
    expected = np.random.random()
    np.random.seed(7)
    options = np.get_printoptions()

    TiccDetector(window_size=3, n_states=2, maxIters=5).fit(_series())

    assert np.random.random() == expected
    assert np.get_printoptions() == options


def test_n_free_parameters_counts_toeplitz_blocks():
    d, w = 2, 3
    theta = np.zeros((d * w, d * w))
    a0 = np.array([[2.0, 0.5], [0.5, 2.0]])
    a1 = np.array([[0.3, 0.0], [0.1, 0.0]])
    for i in range(w):
        theta[i * d : (i + 1) * d, i * d : (i + 1) * d] = a0
    for i in range(w - 1):
        theta[(i + 1) * d : (i + 2) * d, i * d : (i + 1) * d] = a1
        theta[i * d : (i + 1) * d, (i + 1) * d : (i + 2) * d] = a1.T

    # A0: 3 free entries (symmetric); A1: 2 non-zero entries; A2: none
    assert n_free_parameters(theta, w, d) == 5
