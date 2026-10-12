"""Tests of the AutoPlait wrapper's reading of the C binary's output files.

The binary writes one ``segment.<r>`` file per regime ``r``, each line holding
the first and the last index (inclusive) of one segment of that regime, and a
``segment.labels`` file with one row per regime (regime number, path of the
regime in the split tree, cost, number of HMM states). The simulated outputs
below copy the files the binary wrote for the same series; the last test runs
the binary itself when it is built. The binary takes the length of the series
from the number of lines of its input, counted with a 254-byte buffer: the
fake binary counts them the same way.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from tsseg.algorithms.autoplait import autoplait_c
from tsseg.algorithms.autoplait.detector import AutoPlaitDetector
from tsseg.algorithms.utils import extract_cps

USAGE = "error: usage: cmd d fin, fout [./autoplait] \n"


def _rows_counted_by_the_binary(path: Path) -> int:
    """Count the lines of ``path`` as ``count_lines`` of c/autoplait/tool.c does.

    It calls ``fgets`` with a buffer of 254 bytes, which reads at most 253
    characters, the newline included: a longer line counts more than once.
    """
    lines = path.read_text().splitlines()
    return sum(-(-(len(line) + 1) // 253) for line in lines)


def _fake_binary(outputs, stdout: str = ""):
    """Mimic the bundled binary: four arguments only, outputs in the last one.

    ``outputs`` maps file names to their text, or is a function of the input
    file that returns this mapping; the fake binary then reports, as the real
    one does, the number of rows it counts in the input (``duration``).
    """
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(list(command))
        if len(command) != 4:
            return subprocess.CompletedProcess(command, 1, stdout=USAGE, stderr="")
        out_dir = Path(command[-1])
        files, printed = outputs, stdout
        if callable(outputs):
            data_path = Path(Path(command[2]).read_text().split()[0])
            files = outputs(data_path)
            printed = f"duration: {_rows_counted_by_the_binary(data_path)}\n"
        for name, text in files.items():
            (out_dir / name).write_text(text)
        return subprocess.CompletedProcess(command, 0, stdout=printed, stderr="")

    return run, calls


@pytest.fixture
def fake_autoplait(monkeypatch, tmp_path):
    """Install a fake binary; return a function that sets its outputs."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "autoplait").write_text("")
    monkeypatch.setattr(autoplait_c, "_find_autoplait_dir", lambda: bin_dir)

    def install(outputs, stdout: str = ""):
        run, calls = _fake_binary(outputs, stdout)
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

    y = AutoPlaitDetector().fit_predict(X)

    np.testing.assert_array_equal(y, expected)
    assert y.dtype.kind == "i"


def test_change_points_start_the_next_segment(fake_autoplait):
    outputs, expected = CASES["ABCA"]
    fake_autoplait(outputs)

    y = AutoPlaitDetector().fit_predict(np.zeros((len(expected), 1)))

    np.testing.assert_array_equal(extract_cps(y), [300, 600, 900])


def test_tags_declare_an_unsupervised_detector():
    detector = AutoPlaitDetector()

    assert detector.get_tag("capability:unsupervised") is True
    assert detector.get_tag("capability:semi_supervised") is False


def test_runs_without_n_cps_or_y(fake_autoplait):
    outputs, expected = CASES["ABA"]
    calls = fake_autoplait(outputs)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        y = AutoPlaitDetector().fit_predict(np.zeros((900, 1)))

    np.testing.assert_array_equal(y, expected)
    assert [len(command) for command in calls] == [4]


@pytest.mark.parametrize("n_cps", [0, 2])
def test_n_cps_is_deprecated_and_ignored(fake_autoplait, n_cps):
    """A harness in guided mode still passes ``n_cps``: warn, run unguided."""
    outputs, expected = CASES["ABA"]
    calls = fake_autoplait(outputs)

    with pytest.warns(FutureWarning, match="n_cps"):
        y = AutoPlaitDetector(n_cps=n_cps).fit_predict(np.zeros((900, 1)))

    np.testing.assert_array_equal(y, expected)
    assert [len(command) for command in calls] == [4]


@pytest.mark.parametrize(
    "segment_files",
    [
        {"segment.0": "0 299 \n", "segment.1": "300 598 \n"},  # last point missing
        {"segment.0": "0 299 \n", "segment.1": "299 599 \n"},  # point 299 twice
        {"segment.0": "0 299 \n", "segment.1": "300 600 \n"},  # past the end
        {"segment.0": "0 299 \n", "segment.1": "300 599 \n600 650 \n"},  # starts at n
        {"segment.0": "-1 299 \n", "segment.1": "300 599 \n"},  # starts before 0
        {"segment.0": "0 299 \n", "segment.1": "300 200 \n"},  # ends before its start
    ],
)
def test_segments_that_do_not_tile_the_series_raise(fake_autoplait, segment_files):
    fake_autoplait(segment_files)

    with pytest.raises(RuntimeError, match="segment"):
        AutoPlaitDetector().fit_predict(np.zeros((600, 1)))


# Files of c/autoplait/autoplait on ActRecTut subject 1 (31 392 points, channels
# 0-9, as in tsseg-exp) written with ``np.savetxt`` (``%.18e``): the binary
# counted 62 442 lines in its input and segmented that many time points.
ACTRECTUT_1_OUTPUT = {
    "segment.0": "26857 62441 \n",
    "segment.1": "12254 26856 \n",
    "segment.2": "6866 12253 \n",
    "segment.3": "0 6865 \n",
}


def test_a_binary_that_reads_another_length_raises(fake_autoplait):
    fake_autoplait(ACTRECTUT_1_OUTPUT, stdout="duration: 62442\n")

    with pytest.raises(RuntimeError, match="62442 time points instead of 31392"):
        AutoPlaitDetector().fit_predict(np.zeros((31392, 1)))


def _one_segment(data_path: Path) -> dict[str, str]:
    """One regime over every row the binary counts in its input."""
    return {"segment.0": f"0 {_rows_counted_by_the_binary(data_path) - 1} \n"}


def test_the_binary_counts_one_row_per_time_point_on_wide_rows(fake_autoplait):
    """Ten channels of ActRecTut-like values: 259 characters per row in %.18e."""
    fake_autoplait(_one_segment)
    X = np.full((300, 10), -826.6837175792507)

    y = AutoPlaitDetector().fit_predict(X)

    np.testing.assert_array_equal(y, np.zeros(300, dtype=int))


def test_the_input_holds_the_float32_values_the_binary_reads(fake_autoplait):
    written = {}

    def keep_input(data_path):
        written["X"] = np.loadtxt(data_path, dtype=np.float32, ndmin=2)
        return _one_segment(data_path)

    fake_autoplait(keep_input)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 15)) * 10.0 ** rng.integers(-30, 30, (200, 15))

    AutoPlaitDetector().fit_predict(X)

    np.testing.assert_array_equal(written["X"], X.astype(np.float32))


def test_a_row_too_wide_for_the_binary_raises_before_running_it(fake_autoplait):
    calls = fake_autoplait(_one_segment)
    X = np.full((50, 20), -1.23456789e-05)

    with pytest.raises(ValueError, match="20"):
        AutoPlaitDetector().fit_predict(X)

    assert calls == []


def test_missing_segment_files_raise(fake_autoplait):
    fake_autoplait({})

    with pytest.raises(RuntimeError, match="segment"):
        AutoPlaitDetector().fit_predict(np.zeros((600, 1)))


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

    y = AutoPlaitDetector().fit_predict(X)

    np.testing.assert_array_equal(extract_cps(y), [300, 600])
    assert y[0] == y[-1] != y[300]
