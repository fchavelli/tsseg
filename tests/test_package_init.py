"""The top-level ``tsseg`` package.

``import tsseg`` must stay light and free of side effects: it neither patches
NumPy nor imports the detectors, which load on first access to
``tsseg.algorithms``. The top-level ``TGLADDetector`` is a deprecated alias of
``tsseg.algorithms.TGLADDetector``.

The import itself is checked in a fresh interpreter, since this test session
has already imported ``tsseg`` and its subpackages.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import tsseg

_PROBE = textwrap.dedent(
    """
    import json
    import sys

    import tsseg

    report = {
        "numpy_patched": "numpy.core.umath_tests" in sys.modules,
        "algorithms_imported": "tsseg.algorithms" in sys.modules,
        "numba_imported": "numba" in sys.modules,
        "names": sorted(n for n in vars(tsseg) if not n.startswith("_")),
    }
    # Attribute access still reaches the detectors after a bare import.
    report["pelt"] = tsseg.algorithms.PeltDetector.__name__
    print(json.dumps(report))
    """
)


@pytest.fixture(scope="module")
def fresh_import():
    # The subprocess imports the same tsseg as this one.
    root = str(Path(tsseg.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=root,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_import_does_not_patch_numpy(fresh_import):
    # A fake ``numpy.core.umath_tests`` used to be put in ``sys.modules`` for
    # every program that imported tsseg.
    assert not fresh_import["numpy_patched"]


def test_import_does_not_load_the_detectors(fresh_import):
    assert not fresh_import["algorithms_imported"]
    assert not fresh_import["numba_imported"]


def test_import_exposes_only_public_names(fresh_import):
    # Modules imported by ``tsseg/__init__.py`` used to leak as attributes
    # (``tsseg.np``, ``tsseg.sys``...).
    assert fresh_import["names"] == ["data", "load_mocap"]


def test_algorithms_reachable_after_bare_import(fresh_import):
    assert fresh_import["pelt"] == "PeltDetector"


def test_all_lists_load_mocap_only():
    assert tsseg.__all__ == ["load_mocap"]
    namespace = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exec("from tsseg import *", namespace)
    assert sorted(n for n in namespace if not n.startswith("__")) == ["load_mocap"]


def test_top_level_tglad_is_a_deprecated_alias():
    from tsseg.algorithms import TGLADDetector

    with pytest.warns(FutureWarning, match=r"tsseg\.algorithms"):
        alias = tsseg.TGLADDetector
    assert alias is TGLADDetector

    with pytest.warns(FutureWarning, match=r"tsseg\.algorithms"):
        from tsseg import TGLADDetector as Imported
    assert Imported is TGLADDetector


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'NoSuchDetector'"):
        _ = tsseg.NoSuchDetector
    assert not hasattr(tsseg, "NoSuchDetector")
