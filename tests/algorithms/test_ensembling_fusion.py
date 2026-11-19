"""The ensembling strategies fuse the change points channel by channel.

BOCD, BEAST, FLUSS and ChangeFinder run their univariate method on each
channel, then fuse the per-channel change points. The per-channel runner is
replaced by a stub so that the fusion is tested on known channel outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import BOCDDetector, ChangeFinderDetector


def _stub(outputs, kind):
    """Per-channel runner returning ``outputs`` in channel order."""
    calls = iter(outputs)

    def run(self, *args, **kwargs):
        cps = np.asarray(next(calls), dtype=np.int64)
        if kind == "fluss":
            return cps, None, None
        return cps

    return run


def _guided(name, tolerance):
    """Detector in ensembling mode asking for one change point."""
    if name == "bocd":
        detector = BOCDDetector(
            multivariate_strategy="ensembling", max_cps=1, tolerance=tolerance
        )
        return detector, "_run_bocd"
    if name == "beast":
        pytest.importorskip("Rbeast")
        from tsseg.algorithms import BeastDetector

        detector = BeastDetector(
            multivariate_strategy="ensembling", max_cps=1, tolerance=tolerance
        )
        return detector, "_run_beast_univariate"
    if name == "fluss":
        pytest.importorskip("stumpy")
        from tsseg.algorithms import FLUSSDetector

        detector = FLUSSDetector(
            multivariate_strategy="ensembling", n_segments=2, tolerance=tolerance
        )
        return detector, "_run_fluss"
    detector = ChangeFinderDetector(
        multivariate_strategy="ensembling", n_cps=1, tolerance=tolerance
    )
    return detector, "_select_peaks"


def _series(n_channels, n_timepoints=300):
    return np.random.default_rng(0).normal(size=(n_timepoints, n_channels))


@pytest.mark.parametrize("name", ["bocd", "beast", "fluss", "changefinder"])
def test_guided_fusion_counts_channels_not_points(monkeypatch, name):
    # Three change points of channel 0 used to outvote two channels at 100.
    per_channel = [[10, 11, 12], [100], [100]]
    detector, runner = _guided(name, tolerance=4)
    monkeypatch.setattr(type(detector), runner, _stub(per_channel, name))
    np.testing.assert_array_equal(detector.fit_predict(_series(3)), [100])


@pytest.mark.parametrize("name", ["bocd", "beast", "fluss", "changefinder"])
def test_guided_zero_tolerance_is_an_exact_vote(monkeypatch, name):
    per_channel = [[30, 200], [200], [31]]
    detector, runner = _guided(name, tolerance=0)
    monkeypatch.setattr(type(detector), runner, _stub(per_channel, name))
    np.testing.assert_array_equal(detector.fit_predict(_series(3)), [200])


@pytest.mark.parametrize("name", ["fluss", "changefinder"])
def test_unguided_fusion_does_not_chain(monkeypatch, name):
    # Gaps of 4 used to chain 10..22 into a single change point at 16.
    per_channel = [[10], [14], [18], [22]]
    if name == "fluss":
        pytest.importorskip("stumpy")
        from tsseg.algorithms import FLUSSDetector

        detector = FLUSSDetector(
            multivariate_strategy="ensembling", n_segments=None, tolerance=4
        )
        runner = "_run_fluss"
    else:
        detector = ChangeFinderDetector(multivariate_strategy="ensembling", tolerance=4)
        runner = "_select_peaks"
    monkeypatch.setattr(type(detector), runner, _stub(per_channel, name))
    np.testing.assert_array_equal(detector.fit_predict(_series(4)), [14, 22])


def test_bocd_keeps_its_spacing_filter(monkeypatch):
    # Both candidates are elected; the later one is then too close.
    detector = BOCDDetector(
        multivariate_strategy="ensembling", tolerance=0, min_distance=50
    )
    monkeypatch.setattr(BOCDDetector, "_run_bocd", _stub([[100], [100], [130]], ""))
    np.testing.assert_array_equal(detector.fit_predict(_series(3)), [100])
