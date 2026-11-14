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
