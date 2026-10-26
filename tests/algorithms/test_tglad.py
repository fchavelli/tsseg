"""Tests of TGLADDetector: training settings passed to uGLAD (torch required)."""

from __future__ import annotations

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
