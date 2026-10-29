"""How the ``"native"`` strategy of BeastDetector selects its change points.

``beast123`` gives, for each channel, Rbeast's ranked candidates (``cp``) with
their probabilities (``cpPr``). Stand-in outputs need no Rbeast; the last test
runs Rbeast when it is installed.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from tsseg.algorithms import BeastDetector
from tsseg.algorithms.beast import detector as beast_module

NAN = float("nan")


def _use_fake_beast123(monkeypatch, output):
    """Make ``beast123`` return ``output`` (Rbeast need not be installed)."""
    fake = SimpleNamespace(
        beast123=lambda *args, **kwargs: output, args=SimpleNamespace
    )
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: fake)
    # Skip the dependency check at construction, which looks Rbeast up.
    monkeypatch.setattr(
        BeastDetector, "_tags", {**BeastDetector._tags, "python_dependencies": None}
    )


def _component(per_channel):
    """Component with ``cp``/``cpPr`` of shape (slots, channels), NaN-padded.

    ``per_channel`` lists, for each channel, its ``(cp, cpPr)`` pairs in
    Rbeast's order (decreasing ``cpPr``). ``ncp_median`` does not cap them.
    """
    slots = max(len(pairs) for pairs in per_channel) + 1
    cp = np.full((slots, len(per_channel)), NAN)
    pr = np.full((slots, len(per_channel)), NAN)
    for ch, pairs in enumerate(per_channel):
        for i, (c, p) in enumerate(pairs):
            cp[i, ch], pr[i, ch] = c, p
    return SimpleNamespace(
        cp=cp, cpPr=pr, ncp_median=np.full(len(per_channel), float(slots))
    )


def _predict(monkeypatch, output, n_channels=2, n=1000, **params):
    _use_fake_beast123(monkeypatch, output)
    return BeastDetector(**params).fit_predict(np.zeros((n, n_channels))).tolist()


def test_max_cps_keeps_the_most_probable_change_points(monkeypatch):
    """``max_cps`` used to keep the first candidates by position."""
    output = SimpleNamespace(
        trend=_component([[(400, 0.95), (700, 0.5)], [(150, 0.3)]])
    )
    assert _predict(monkeypatch, output) == [150, 400, 700]
    assert _predict(monkeypatch, output, max_cps=1) == [400]
    assert _predict(monkeypatch, output, max_cps=2) == [400, 700]
    assert _predict(monkeypatch, output, max_cps=3) == [150, 400, 700]
    assert _predict(monkeypatch, output, max_cps=0) == []


def test_a_candidate_found_on_several_channels_has_its_largest_probability(
    monkeypatch,
):
    output = SimpleNamespace(trend=_component([[(500, 0.4), (600, 0.2)], [(600, 0.9)]]))
    assert _predict(monkeypatch, output, max_cps=1) == [600]


def test_max_cps_ranks_the_seasonal_candidates_by_their_own_probability(
    monkeypatch,
):
    output = SimpleNamespace(
        trend=_component([[(200, 0.3)], [(250, 0.2)]]),
        season=_component([[(700, 0.9)], []]),
    )
    assert _predict(monkeypatch, output, component="both") == [200, 250, 700]
    assert _predict(monkeypatch, output, component="both", max_cps=1) == [700]
    assert _predict(monkeypatch, output, component="season", max_cps=1) == [700]
    assert _predict(monkeypatch, output, component="trend", max_cps=1) == [200]


def test_tolerance_merges_close_candidates(monkeypatch):
    """The same change seen by two channels, one point apart, counts once."""
    output = SimpleNamespace(
        trend=_component([[(400, 0.9), (150, 0.5)], [(401, 0.8), (153, 0.7)]])
    )
    # tolerance=0 (the default) merges nothing: the union of the channels
    assert _predict(monkeypatch, output) == [150, 153, 400, 401]
    # a candidate within tolerance of a more probable one is dropped
    assert _predict(monkeypatch, output, tolerance=1) == [150, 153, 400]
    assert _predict(monkeypatch, output, tolerance=3) == [153, 400]
    # merging comes before max_cps, which then keeps two distinct changes
    assert _predict(monkeypatch, output, max_cps=2) == [400, 401]
    assert _predict(monkeypatch, output, tolerance=3, max_cps=2) == [153, 400]
    # a float below 1 is a fraction of the series length, as with "ensembling"
    assert _predict(monkeypatch, output, tolerance=0.002) == [150, 153, 400]
    assert _predict(monkeypatch, output, tolerance=0.002, n=2000) == [153, 400]


def test_tolerance_does_not_chain_merges(monkeypatch):
    """Each candidate is compared with the candidates kept, not with its neighbours."""
    output = SimpleNamespace(trend=_component([[(300, 0.9), (310, 0.5)], [(305, 0.6)]]))
    assert _predict(monkeypatch, output, tolerance=5) == [300, 310]


def test_native_max_cps_on_a_real_series():
    """A weak change on one channel, a strong one on the other: keep the strong."""
    pytest.importorskip("Rbeast")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (600, 2))
    X[150:, 1] += 1.0
    X[400:, 0] += 4.0
    every = BeastDetector().fit_predict(X)
    assert len(every) > 1
    capped = BeastDetector(max_cps=1).fit_predict(X)
    assert len(capped) == 1
    assert abs(int(capped[0]) - 400) <= 3, capped
