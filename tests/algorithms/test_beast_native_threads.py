"""BeastDetector ``"native"`` and the threads of Rbeast's ``beast123``.

Given a 2D array, Rbeast 0.1.25 fits its columns independently, in threads
pinned to CPUs 0, 1, ... (``glue_code.c``) whatever CPUs the process may use.
Where it may not use them (a job of a cluster scheduler, a container limited
to some CPUs), the threads are not created: Rbeast returns unset outputs and
corrupts the memory. ``"native"`` therefore runs ``beast123`` on one channel
at a time, which Rbeast computes in the calling thread.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tsseg.algorithms import BeastDetector
from tsseg.algorithms.beast import detector as beast_module

NAN = float("nan")
ROOT = Path(__file__).resolve().parents[2]


def _step(n=1000, n_channels=3):
    """Every channel steps up at n // 2; channel k has its own noise."""
    x = np.zeros((n, n_channels))
    x[n // 2 :] += 3.0
    for k in range(n_channels):
        x[:, k] += np.random.default_rng(k).normal(0, 0.3, n)
    return x


def test_native_passes_one_channel_at_a_time_to_beast123(monkeypatch):
    """It used to pass the whole array, which Rbeast splits among threads."""
    received = []

    def beast123(data, **kwargs):
        received.append(np.array(data))
        trend = SimpleNamespace(
            cp=np.array([500.0, NAN]),
            cpPr=np.array([0.9, NAN]),
            ncp_median=np.array([1.0]),
        )
        return SimpleNamespace(trend=trend)

    fake = SimpleNamespace(beast123=beast123, args=SimpleNamespace)
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: fake)
    monkeypatch.setattr(
        BeastDetector, "_tags", {**BeastDetector._tags, "python_dependencies": None}
    )
    X = _step()
    assert BeastDetector().fit_predict(X).tolist() == [500]
    assert [data.shape for data in received] == [(1000,)] * 3
    for k, data in enumerate(received):
        np.testing.assert_array_equal(data, X[:, k])


def _run_in_child(code):
    """Run ``code`` in a fresh interpreter: a corrupted heap kills the child."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=ROOT,
    )


def _allowed_cpus():
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("os.sched_getaffinity is not available")
    return os.sched_getaffinity(0)


_NATIVE_ON_A_STEP = """
import numpy as np
from tsseg.algorithms import BeastDetector

x = np.zeros((1000, {n_channels}))
x[500:] += 3.0
for k in range({n_channels}):
    x[:, k] += np.random.default_rng(k).normal(0, 0.3, 1000)
print(BeastDetector().fit_predict(x).tolist())
for _ in range(20):  # churn the heap: a corruption shows here
    blocks = [np.random.rand(1000, 50) for _ in range(50)]
    del blocks
"""


@pytest.mark.parametrize("n_channels", [2, 4])
def test_native_where_the_process_may_not_use_the_first_cpus(n_channels):
    """In a job limited to CPUs other than 0, 1, ...: no CP, wrong ones, or a crash.

    Runs only where the process may not use one of the CPUs ``beast123``
    would pin its threads to (e.g. a cluster job); elsewhere ``"native"``
    worked before as after.
    """
    pytest.importorskip("Rbeast")
    if set(range(n_channels)) <= _allowed_cpus():
        pytest.skip(f"the process may use CPUs 0 to {n_channels - 1}")
    child = _run_in_child(_NATIVE_ON_A_STEP.format(n_channels=n_channels))
    assert child.returncode == 0, (child.returncode, child.stderr[-2000:])
    assert child.stdout.split() == ["[500]"], child.stdout


_PER_CHANNEL_VERSUS_WHOLE = """
import numpy as np
import Rbeast

from tsseg.algorithms import BeastDetector

real = Rbeast.beast123
runs = []


def spy(data, **kwargs):
    runs.append((np.array(data), kwargs, real(data, **kwargs)))
    return runs[-1][2]


Rbeast.beast123 = spy
rng = np.random.default_rng(0)
x = rng.normal(0, 0.3, (800, 3))
x[300:, 0] += 2.0
x[500:, 1] -= 2.0
x[200:, 2] += 1.5
BeastDetector().fit_predict(x)
assert len(runs) == 3, len(runs)
whole = real(np.column_stack([run[0] for run in runs]), **runs[0][1])
for ch, (_, _, single) in enumerate(runs):
    for field in ("cp", "cpPr", "ncp_median"):
        a = np.asarray(getattr(whole.trend, field), dtype=float)
        a = a[:, ch] if a.ndim == 2 else a.ravel()[ch : ch + 1]
        b = np.asarray(getattr(single.trend, field), dtype=float).ravel()
        np.testing.assert_array_equal(a, b, err_msg=f"channel {ch}, {field}")
print("same")
"""


def test_one_run_per_channel_gives_the_run_on_the_whole_array():
    """Where Rbeast's threads can be created, ``"native"`` gives what it gave.

    Each column of a 2D array is fitted in its own thread, with the same seed
    as a single series: one ``beast123`` run per channel returns the same
    change points, probabilities and ``ncp_median``.
    """
    pytest.importorskip("Rbeast")
    if not set(range(3)) <= _allowed_cpus():
        pytest.skip("the process may not use CPUs 0 to 2: Rbeast's threads fail")
    child = _run_in_child(_PER_CHANNEL_VERSUS_WHOLE)
    assert child.returncode == 0, (child.returncode, child.stderr[-2000:])
    assert child.stdout.split() == ["same"], child.stdout
