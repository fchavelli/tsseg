"""Tests of TGLADDetector: training settings passed to uGLAD, predict on a new series (torch required)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("networkx")

from tsseg.algorithms import TGLADDetector  # noqa: E402
from tsseg.algorithms.tglad.vendor import ensure_vendor_imports  # noqa: E402


@pytest.fixture
def uglad_calls(monkeypatch):
    """Replace the uGLAD solver by a stub that records its arguments.

    The stub returns an identity precision matrix per window, so that
    ``fit`` runs without training anything.
    """
    ensure_vendor_imports()
    from tsseg.algorithms.tglad.vendor.uGLAD import uglad

    calls: list[dict] = []

    def fake_fit(self, Xb, **kwargs):
        calls.append(kwargs)
        self.precision_ = [np.eye(np.asarray(X).shape[1]) for X in Xb]
        return self

    monkeypatch.setattr(uglad.uGLAD_multitask, "fit", fake_fit)
    return calls


def _series() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(400, 2))


def test_default_epochs_match_the_authors_code(uglad_calls):
    """2000 epochs by default, as in ``main.py`` of the tGLAD repository.

    With 100 epochs, the soft threshold that GLAD learns can set every
    partial correlation of a multivariate window to zero, and then no
    change point is ever emitted.
    """
    detector = TGLADDetector(window_size=100, stride=50)
    assert detector.epochs == 2000
    detector.fit(_series())
    assert uglad_calls
    assert all(call["epochs"] == 2000 for call in uglad_calls)
    assert all(call["lr"] == 0.001 and call["L"] == 5 for call in uglad_calls)


def test_epochs_reach_the_solver(uglad_calls):
    TGLADDetector(window_size=100, stride=50, batch_size=2, epochs=100).fit(_series())
    # Seven windows in batches of two: four calls.
    assert len(uglad_calls) == 4
    assert all(call["epochs"] == 100 for call in uglad_calls)


def test_fit_prints_nothing_by_default(capsys):
    """``verbose=False`` (the default) keeps uGLAD quiet on stdout.

    The vendored ``uGLAD_multitask.fit`` used to print three lines per
    batch whatever ``verbose`` said ("Running uGLAD in multi-task mode",
    "Processing the input table...", "Total runtime...").
    """
    TGLADDetector(window_size=100, stride=50, epochs=20).fit(_series())
    assert capsys.readouterr().out == ""


def test_verbose_prints_the_progress(capsys):
    TGLADDetector(window_size=100, stride=50, epochs=20, verbose=True).fit(_series())
    out = capsys.readouterr().out
    assert "Running uGLAD in multi-task mode" in out
    assert "epoch:0/20" in out
    assert "Total runtime" in out


@pytest.mark.parametrize("verbose", [False, True])
def test_fewer_than_ten_epochs(verbose, capsys):
    """The schema accepts ``epochs >= 1``; uGLAD printed every
    ``int(epochs / 10)`` epochs, a modulo by zero below ten epochs."""
    detector = TGLADDetector(window_size=100, stride=50, epochs=5, verbose=verbose)
    detector.fit(_series())
    assert detector.change_scores_.shape == (7,)
    out = capsys.readouterr().out
    assert ("epoch:0/5" in out) is verbose


# predict() on a series other than the fitted one: it returned the change
# points of fit whatever X.

REFIT_WARNING = "differs from that given to fit"
WINDOWS = {"window_size": 100, "stride": 50}


@pytest.fixture
def correlation_precisions(monkeypatch):
    """Replace the uGLAD solver by the inverse correlation matrix of each window.

    Deterministic and fast, unlike uGLAD, and the change points still follow
    the changes of correlation of the series. Returns the list of the windows
    given to the solver.
    """
    ensure_vendor_imports()
    from tsseg.algorithms.tglad.vendor.uGLAD import uglad

    windows: list[np.ndarray] = []

    def fake_fit(self, Xb, **kwargs):
        windows.extend(Xb)
        self.precision_ = [np.linalg.inv(np.corrcoef(np.asarray(X).T)) for X in Xb]
        return self

    monkeypatch.setattr(uglad.uGLAD_multitask, "fit", fake_fit)
    return windows


def _correlated(*rhos: float) -> np.ndarray:
    """Two channels, 200 points per correlation coefficient in ``rhos``."""
    rng = np.random.default_rng(0)
    parts = []
    for rho in rhos:
        z = rng.normal(size=(200, 2))
        parts.append(np.c_[z[:, 0], rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]])
    return np.concatenate(parts)


def test_predict_on_another_series_of_the_same_length(correlation_precisions):
    x1, x2 = _correlated(0.9, -0.9), _correlated(-0.9, 0.9)
    expected = TGLADDetector(**WINDOWS).fit_predict(x2)
    detector = TGLADDetector(**WINDOWS).fit(x1)
    fitted, scores = detector.change_points_, detector.change_scores_
    assert not np.array_equal(expected, fitted)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x2), expected)
    # the fitted attributes still describe x1
    np.testing.assert_array_equal(detector.change_points_, fitted)
    np.testing.assert_array_equal(detector.change_scores_, scores)


def test_predict_on_a_shorter_series(correlation_precisions):
    # predict() used to return [100, 150, 200] on a series of 200 points
    x1 = _correlated(0.9, -0.9)
    expected = TGLADDetector(**WINDOWS).fit_predict(x1[:200])
    detector = TGLADDetector(**WINDOWS).fit(x1)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        cps = detector.predict(x1[:200])
    np.testing.assert_array_equal(cps, expected)
    assert np.all(cps < 200)


def test_predict_after_the_fit_series_is_modified_in_place(correlation_precisions):
    x = _correlated(0.9, -0.9)
    expected = TGLADDetector(**WINDOWS).fit_predict(_correlated(-0.9, 0.9))
    detector = TGLADDetector(**WINDOWS).fit(x)
    x[:] = _correlated(-0.9, 0.9)
    with pytest.warns(UserWarning, match=REFIT_WARNING):
        np.testing.assert_array_equal(detector.predict(x), expected)


def test_predict_on_the_fit_series_reuses_the_fit(correlation_precisions):
    x = _correlated(0.9, -0.9)
    detector = TGLADDetector(**WINDOWS).fit(x)
    n_windows = len(correlation_precisions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # equal values in a new array are the series given to fit
        cps = detector.predict(x.copy())
    np.testing.assert_array_equal(cps, detector.change_points_)
    # no new training
    assert len(correlation_precisions) == n_windows
