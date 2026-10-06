"""ClaSP under a numba thread limit lower than the number of cores.

``numba.set_num_threads`` refuses more threads than ``NUMBA_NUM_THREADS``,
which users and schedulers (e.g. SLURM jobs) set below the number of cores.
ClaSP resolved ``n_jobs=-1`` to ``os.cpu_count()`` and failed with
``ValueError: The number of threads must be between 1 and ...``.
``NUMBA_NUM_THREADS`` is read when numba is imported, hence the subprocesses.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from tsseg.algorithms import ClaspDetector

    t = np.arange(400)
    x = np.concatenate([np.sin(t / 4), np.sign(np.sin(t / 9))])  # shape change at 400
    detector = ClaspDetector(n_change_points=1, window_size=20, n_jobs={n_jobs})
    print(detector.fit_predict(x).tolist())
    """
)


def _run(n_jobs: int, numba_threads: int) -> subprocess.CompletedProcess:
    env = dict(os.environ, NUMBA_NUM_THREADS=str(numba_threads))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-c", SCRIPT.format(n_jobs=n_jobs)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        timeout=600,
    )


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="needs more than one core")
@pytest.mark.parametrize("n_jobs", [-1, 4])
def test_clasp_runs_below_the_core_count(n_jobs):
    result = _run(n_jobs=n_jobs, numba_threads=1)
    assert result.returncode == 0, result.stderr[-2000:]
    change_points = ast.literal_eval(result.stdout.strip().splitlines()[-1])
    assert len(change_points) == 1
    assert abs(change_points[0] - 400) <= 20


def test_resolve_n_jobs_caps_at_the_numba_limit():
    import numba

    from tsseg.algorithms.clap.utils import resolve_n_jobs

    limit = numba.config.NUMBA_NUM_THREADS
    assert resolve_n_jobs(-1) == limit
    assert resolve_n_jobs(0) == limit
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(limit + 5) == limit
