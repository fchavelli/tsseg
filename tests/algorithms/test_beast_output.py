"""How BeastDetector reads Rbeast outputs (stand-in outputs, no Rbeast needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from tsseg.algorithms import BeastDetector
from tsseg.algorithms.beast import detector as beast_module

NAN = float("nan")


def _fake_rbeast(output):
    """Stand-in for the Rbeast module whose ``beast``/``beast123`` return ``output``."""

    def run(*args, **kwargs):
        return output

    return SimpleNamespace(beast=run, beast123=run, args=SimpleNamespace)


def _trend(cp, cp_pr, ncp_median):
    return SimpleNamespace(
        cp=np.array(cp, dtype=float),
        cpPr=np.array(cp_pr, dtype=float),
        ncp_median=np.array(ncp_median, dtype=float),
        cpOccPr=np.zeros(1000),
    )


def test_ncp_median_caps_the_ranked_change_points(monkeypatch):
    output = SimpleNamespace(trend=_trend([300, 700, NAN], [0.9, 0.8, NAN], [1.0]))
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: _fake_rbeast(output))
    assert BeastDetector().fit_predict(np.zeros(1000)).tolist() == [300]


def test_nan_ncp_median_does_not_crash_univariate(monkeypatch):
    """Rbeast can return NaN for ``ncp_median`` (seen in CI on 6,000 points)."""
    output = SimpleNamespace(trend=_trend([300, 700, NAN], [0.9, 0.8, NAN], [NAN]))
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: _fake_rbeast(output))
    with pytest.warns(RuntimeWarning, match="ncp_median"):
        cps = BeastDetector().fit_predict(np.zeros(1000))
    assert cps.tolist() == [300, 700]


def test_nan_ncp_median_does_not_crash_multivariate(monkeypatch):
    output = SimpleNamespace(
        trend=_trend(
            [[300, 300], [700, 700], [NAN, NAN]],
            [[0.9, 0.9], [0.8, 0.8], [NAN, NAN]],
            [1.0, NAN],
        )
    )
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: _fake_rbeast(output))
    with pytest.warns(RuntimeWarning, match="ncp_median"):
        cps = BeastDetector().fit_predict(np.zeros((1000, 2)))
    # Channel 0 is capped at one CP by its ncp_median, channel 1 is not.
    assert cps.tolist() == [300, 700]


def test_all_nan_output_gives_no_change_point(monkeypatch):
    """A failed Rbeast run returns NaN everywhere: no CP, and a warning."""
    output = SimpleNamespace(trend=_trend([NAN, NAN], [NAN, NAN], [NAN]))
    output.trend.cpOccPr = np.full(1000, NAN)
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: _fake_rbeast(output))
    with pytest.warns(RuntimeWarning, match="ncp_median"):
        cps = BeastDetector().fit_predict(np.zeros(1000))
    assert cps.size == 0
    assert cps.dtype == np.int64
