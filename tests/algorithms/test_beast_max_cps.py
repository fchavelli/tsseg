"""How ``BeastDetector`` ranks the change points it keeps under ``max_cps``.

Univariate path (also used by the ``"l2"`` and ``"ensembling"`` strategies):
each candidate is ranked by the occurrence probability (``cpOccPr``) at its
index of the returned component (``component``), the larger of the two with
``"both"``. The stand-in outputs below need no Rbeast; the last
test runs Rbeast on a seasonal series, in a child process because a seasonal
model can kill the Rbeast process (see the ``season`` parameter).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tsseg
from tsseg.algorithms import BeastDetector
from tsseg.algorithms.beast import detector as beast_module

N = 480
NAN = float("nan")


def _component(cp, cp_pr, occ):
    """A Rbeast output component: ranked ``cp``/``cpPr`` and dense ``cpOccPr``."""
    cp_occ = np.zeros(N)
    for index, prob in occ.items():
        cp_occ[index] = prob
    return SimpleNamespace(
        cp=np.array([*cp, NAN], dtype=float),
        cpPr=np.array([*cp_pr, NAN], dtype=float),
        ncp_median=np.array([float(len(cp))]),
        cpOccPr=cp_occ,
    )


def _use_fake_rbeast(monkeypatch, output):
    """Stand in for Rbeast, which need not be installed."""
    fake = SimpleNamespace(beast=lambda *args, **kwargs: output)
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: fake)
    # Skip the dependency check at construction, which looks Rbeast up.
    monkeypatch.setattr(
        BeastDetector, "_tags", {**BeastDetector._tags, "python_dependencies": None}
    )


def _output():
    """A trend change at 100, and two seasonal changes, the surer one at 360.

    The trend's occurrence probability is high at 200 and nil at 360: ranking
    the seasonal candidates by it would keep 200.
    """
    trend = _component([100], [0.7], {100: 0.7, 200: 0.8})
    season = _component([360, 200], [1.0, 0.6], {360: 1.0, 200: 0.6, 100: 0.9})
    return SimpleNamespace(trend=trend, season=season)


def _predict(monkeypatch, output, **params):
    _use_fake_rbeast(monkeypatch, output)
    detector = BeastDetector(season="harmonic", period=24, **params)
    return detector.fit_predict(np.zeros(N)).tolist()


def test_season_candidates_ranked_by_season_probability(monkeypatch):
    output = _output()
    assert _predict(monkeypatch, output, component="season") == [200, 360]
    assert _predict(monkeypatch, output, component="season", max_cps=1) == [360]


def test_both_components_ranked_by_the_larger_probability(monkeypatch):
    output = _output()
    assert _predict(monkeypatch, output, component="both") == [100, 200, 360]
    assert _predict(monkeypatch, output, component="both", max_cps=1) == [360]
    # 100 (season 0.9) and 360 (season 1.0) beat 200 (trend 0.8, season 0.6).
    assert _predict(monkeypatch, output, component="both", max_cps=2) == [100, 360]


def test_trend_candidates_still_ranked_by_trend_probability(monkeypatch):
    trend = _component([100, 200], [0.9, 0.8], {100: 0.4, 200: 0.8})
    season = _component([300], [1.0], {100: 1.0, 300: 1.0})
    output = SimpleNamespace(trend=trend, season=season)
    # The season's probability at 100 does not count for the trend.
    assert _predict(monkeypatch, output, component="trend", max_cps=1) == [200]


_CHILD = """
import json
import numpy as np
from tsseg.algorithms import BeastDetector

n, period = 480, 24
rng = np.random.default_rng(100)
t = np.arange(n)
amplitude = np.ones(n)
for start, value in [(120, 1.3), (240, 1.6), (360, 4.0)]:
    amplitude[start:] = value
x = amplitude * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n)
detector = BeastDetector(
    season="harmonic", period=period, component="season", max_cps=1
)
print(json.dumps(detector.fit_predict(x).tolist()))
"""


def test_seasonal_series_keeps_the_strongest_seasonal_change():
    """Three amplitude changes; the one at 360 (amplitude 1.6 -> 4.0) is surest.

    Ranked by the trend's occurrence probability, ``max_cps=1`` kept 203.
    """
    pytest.importorskip("Rbeast")
    root = Path(tsseg.__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root), *sys.path])}
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=600,
    )
    if result.returncode < 0:  # killed by a signal: a Rbeast crash, not a result
        pytest.skip(f"Rbeast killed the child process (signal {-result.returncode})")
    assert result.returncode == 0, result.stderr
    (cp,) = json.loads(result.stdout.strip().splitlines()[-1])
    assert abs(cp - 360) <= 5
