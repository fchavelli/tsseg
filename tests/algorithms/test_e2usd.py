"""Behaviour of E2USDDetector not covered by the generic suite."""

import numpy as np
import pytest

pytest.importorskip("torch")

from tsseg.algorithms import E2USDDetector  # noqa: E402

SMALL = {
    "window_size": 32,
    "step": 16,
    "n_states": 2,
    "nb_steps": 2,
    "random_state": 0,
}


class _SplitClustering:
    """Stand-in for the DPGMM: the second half of the windows is state 1."""

    def fit(self, X):
        n_windows = len(X)
        return (np.arange(n_windows) >= n_windows // 2).astype(int)


def _two_regime_series(n_per=150, rng=0):
    rng = np.random.default_rng(rng)
    return np.concatenate([rng.normal(0.0, 0.3, n_per), rng.normal(2.0, 0.3, n_per)])


def test_fit_predict_writes_nothing_to_stdout(capsys):
    labels = E2USDDetector(**SMALL).fit_predict(_two_regime_series())
    assert labels.shape == (300,)
    assert capsys.readouterr().out == ""


def test_predict_writes_nothing_to_stdout(capsys):
    detector = E2USDDetector(**SMALL).fit(_two_regime_series())
    capsys.readouterr()
    detector.predict(_two_regime_series(rng=1))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("n_timepoints", "step"),
    [(300, 20), (292, 20), (300, 30), (301, 3)],
    ids=["tail-8", "no-tail", "tail-28", "tail-2"],
)
def test_points_after_the_last_window_take_its_label(n_timepoints, step):
    window_size = SMALL["window_size"]
    X = _two_regime_series(n_per=151)[:n_timepoints]
    detector = E2USDDetector(**{**SMALL, "step": step}).fit(X)
    detector.clustering_ = _SplitClustering()

    labels = detector.predict(X)

    n_windows = (n_timepoints - window_size) // step + 1
    covered = (n_windows - 1) * step + window_size
    assert covered == n_timepoints - (n_timepoints - window_size) % step
    assert labels.shape == (n_timepoints,)
    assert labels[0] == 0
    # The last window, labelled 1, alone covers the last covered point.
    assert labels[covered - 1] == 1
    # The points that no window covers extend the vote of the last window.
    np.testing.assert_array_equal(labels[covered:], 1)
