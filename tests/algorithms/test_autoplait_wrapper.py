"""Tests of the AutoPlait wrapper's reading of the C binary's output files.

The binary writes one ``segment.<r>`` file per regime ``r``, each line holding
the first and the last index (inclusive) of one segment of that regime, and a
``segment.labels`` file with one row per regime (regime number, path of the
regime in the split tree, cost, number of HMM states). The simulated outputs
below copy the files the binary wrote for the same series; the last test runs
the binary itself when it is built.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from tsseg.algorithms.autoplait import autoplait_c
from tsseg.algorithms.autoplait.detector import AutoPlaitDetector
from tsseg.algorithms.utils import extract_cps

USAGE = "error: usage: cmd d fin, fout [./autoplait] \n"


def _fake_binary(outputs: dict[str, str]):
    """Mimic the bundled binary: four arguments only, outputs in the last one."""
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        if len(command) != 4:
            return subprocess.CompletedProcess(command, 1, stdout=USAGE, stderr="")
        out_dir = Path(command[-1])
        for name, text in outputs.items():
            (out_dir / name).write_text(text)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run, calls


@pytest.fixture
def fake_autoplait(monkeypatch, tmp_path):
    """Install a fake binary; return a function that sets its outputs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "autoplait").write_text("")
    monkeypatch.setattr(autoplait_c, "_find_autoplait_dir", lambda: bin_dir)

    def install(outputs: dict[str, str]):
        run, calls = _fake_binary(outputs)
        monkeypatch.setattr(autoplait_c.subprocess, "run", run)
        return calls

    return install


def _labels_row(regime: int, path: str, cost: int, k: int = 1) -> str:
    return f"{regime}\t\t{path}\t\t{cost}\t\t{k} \n"


# Outputs of c/autoplait/autoplait on Gaussian segments of 300 points.
CASES = {
    # A-B-A: regime 0 = A (two segments), regime 1 = B.
    "ABA": (
        {
            "segment.0": "0 299 \n600 899 \n",
            "segment.1": "300 599 \n",
            "segment.labels": _labels_row(0, "1", 563) + _labels_row(1, "0", 346),
        },
        [0] * 300 + [1] * 300 + [0] * 300,
    ),
    # A-B-C-A: regime 1 = A; split-tree paths "01" and "00" for regimes 1, 2.
    "ABCA": (
        {
            "segment.0": "600 899 \n",
            "segment.1": "0 299 \n900 1199 \n",
            "segment.2": "300 599 \n",
            "segment.labels": (
                _labels_row(0, "1", 261)
                + _labels_row(1, "01", 402)
                + _labels_row(2, "00", 272)
            ),
        },
        [1] * 300 + [2] * 300 + [0] * 300 + [1] * 300,
    ),
    # White noise: one regime, empty split-tree path.
    "single_regime": (
        {"segment.0": "0 899 \n", "segment.labels": "0\t\t\t\t1992\t\t1 \n"},
        [0] * 900,
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_labels_follow_the_regime_files(fake_autoplait, case):
    outputs, expected = CASES[case]
    fake_autoplait(outputs)
    X = np.zeros((len(expected), 2))

    y = AutoPlaitDetector(n_cps=1).fit_predict(X)

    np.testing.assert_array_equal(y, expected)
    assert y.dtype.kind == "i"


def test_change_points_start_the_next_segment(fake_autoplait):
    outputs, expected = CASES["ABCA"]
    fake_autoplait(outputs)

    y = AutoPlaitDetector(n_cps=3).fit_predict(np.zeros((len(expected), 1)))

    np.testing.assert_array_equal(extract_cps(y), [300, 600, 900])


def test_n_cps_is_tried_then_the_binary_runs_unguided(fake_autoplait):
    calls = fake_autoplait(CASES["ABA"][0])

    AutoPlaitDetector(n_cps=2).fit_predict(np.zeros((900, 1)))

    assert [len(command) for command in calls] == [5, 4]
    assert calls[0][2] == "3"


@pytest.mark.parametrize(
    "segment_files",
    [
        {"segment.0": "0 299 \n", "segment.1": "300 598 \n"},  # last point missing
        {"segment.0": "0 299 \n", "segment.1": "299 599 \n"},  # point 299 twice
        {"segment.0": "0 299 \n", "segment.1": "300 600 \n"},  # past the end
    ],
)
def test_segments_that_do_not_tile_the_series_raise(fake_autoplait, segment_files):
    fake_autoplait(segment_files)

    with pytest.raises(RuntimeError, match="segment"):
        AutoPlaitDetector(n_cps=1).fit_predict(np.zeros((600, 1)))


def test_missing_segment_files_raise(fake_autoplait):
    fake_autoplait({})

    with pytest.raises(RuntimeError, match="segment"):
        AutoPlaitDetector(n_cps=1).fit_predict(np.zeros((600, 1)))


def _binary_available() -> bool:
    try:
        return (autoplait_c._find_autoplait_dir() / "autoplait").is_file()
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _binary_available(), reason="AutoPlait C binary not built")
def test_binary_on_a_repeated_regime():
    rng = np.random.default_rng(1)
    X = np.concatenate(
        [rng.normal(0, 1, 300), rng.normal(5, 1, 300), rng.normal(0, 1, 300)]
    )

    y = AutoPlaitDetector(n_cps=2).fit_predict(X)

    np.testing.assert_array_equal(extract_cps(y), [300, 600])
    assert y[0] == y[-1] != y[300]
