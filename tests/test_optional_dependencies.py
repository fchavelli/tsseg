"""Detectors whose optional dependencies are missing.

Every name of ``tsseg.algorithms.__all__`` must import without its optional
dependencies, so that ``from tsseg.algorithms import *`` and ``hasattr`` work
on a bare ``pip install tsseg``. A detector whose dependencies are missing
fails at construction, with a message that names the extra to install.

The missing dependencies are simulated in a subprocess, by a finder that
refuses to import them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import tsseg
from tsseg.algorithms.base import BaseSegmenter

#: Top-level modules of every optional dependency of a detector.
OPTIONAL_MODULES = sorted(
    {
        "Rbeast",
        "aeon",
        "changeforest",
        "networkx",
        "npbad",
        "numba",
        "prophet",
        "stumpy",
        "tcn",
        "tensorflow",
        "tensorflow_probability",
        "tf_keras",
        "torch",
    }
)

_PROBE = textwrap.dedent(
    """
    import json
    import sys

    BLOCKED = set(sys.argv[1].split(","))


    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None


    sys.meta_path.insert(0, _Blocker())

    import tsseg.algorithms as algorithms

    namespace = {}
    exec("from tsseg.algorithms import *", namespace)
    report = {"star": sorted(n for n in namespace if not n.startswith("__"))}
    for name in algorithms.__all__:
        entry = {"hasattr": hasattr(algorithms, name)}
        cls = getattr(algorithms, name)
        deps = cls.get_class_tag("python_dependencies") or []
        entry["deps"] = list(deps)
        if set(deps) & BLOCKED:
            try:
                cls()
            except ModuleNotFoundError as exc:
                entry["error"] = [type(exc).__name__, str(exc), exc.name]
            else:
                entry["error"] = None
        report[name] = entry
    print(json.dumps(report))
    """
)


@pytest.fixture(scope="module")
def report():
    # The subprocess imports the same tsseg as this one.
    root = str(Path(tsseg.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([root, os.environ.get("PYTHONPATH", "")]),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, ",".join(OPTIONAL_MODULES)],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=root,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _detectors():
    import tsseg.algorithms as algorithms

    return list(algorithms.__all__)


def test_star_import_without_optional_dependencies(report):
    assert report["star"] == sorted(_detectors())


@pytest.mark.parametrize("name", _detectors())
def test_detector_imports_without_optional_dependencies(report, name):
    assert report[name]["hasattr"]


@pytest.mark.parametrize("name", _detectors())
def test_construction_names_the_missing_extra(report, name):
    deps = set(report[name]["deps"])
    if not deps:
        pytest.skip(f"{name} has no optional dependency")
    assert deps <= set(OPTIONAL_MODULES), "add the module to OPTIONAL_MODULES"
    assert report[name]["error"] is not None, f"{name}() did not raise"
    kind, message, missing = report[name]["error"]
    assert kind == "ModuleNotFoundError"
    assert message.startswith(f"{name} requires ")
    assert missing in deps
    if deps != {"npbad"}:  # not on PyPI, hence in no extra
        assert re.search(r'pip install "tsseg\[[a-z0-9,]+\]"', message), message


class _NeedsNothing(BaseSegmenter):
    _tags = {"python_dependencies": ["numpy"]}

    def __init__(self, axis=0):
        super().__init__(axis=axis)

    def _predict(self, X):
        return np.array([], dtype=int)


class _NeedsMissing(BaseSegmenter):
    _tags = {"python_dependencies": ["numpy", "tsseg_no_such_module"]}

    def __init__(self, axis=0):
        super().__init__(axis=axis)

    def _predict(self, X):
        return np.array([], dtype=int)


def test_installed_dependencies_pass():
    from tsseg.algorithms._dependencies import missing_dependencies

    assert missing_dependencies(["numpy", "scipy"]) == []
    _NeedsNothing()


def test_missing_dependency_raises_at_construction():
    with pytest.raises(ModuleNotFoundError) as info:
        _NeedsMissing()
    assert info.value.name == "tsseg_no_such_module"
    assert str(info.value) == (
        "_NeedsMissing requires tsseg_no_such_module, which is not installed."
    )


@pytest.mark.parametrize(
    ("deps", "missing", "command"),
    [
        (["numba", "aeon"], ["numba", "aeon"], "tsseg[accelerators,aeon]"),
        (["numba", "aeon"], ["aeon"], "tsseg[aeon]"),
        (["torch"], ["torch"], "tsseg[torch]"),
        (["torch", "networkx"], ["torch", "networkx"], "tsseg[tglad]"),
        (["torch", "networkx"], ["torch"], "tsseg[torch]"),
        (["tensorflow", "tcn"], ["tensorflow"], "tsseg[tscp2]"),
        (
            ["tensorflow", "tensorflow_probability", "tf_keras"],
            ["tensorflow"],
            "tsseg[snlds]",
        ),
    ],
)
def test_message_names_the_smallest_extras(deps, missing, command, monkeypatch):
    import tsseg.algorithms._dependencies as module

    monkeypatch.setattr(module, "missing_dependencies", lambda _: list(missing))

    class Foo(BaseSegmenter):
        _tags = {"python_dependencies": deps}

    with pytest.raises(ModuleNotFoundError) as info:
        module.check_python_dependencies(Foo)
    assert (
        str(info.value) == f'Foo requires {", ".join(missing)}: pip install "{command}"'
    )


def test_extras_only_name_optional_modules():
    from tsseg.algorithms._dependencies import EXTRAS

    assert set().union(*EXTRAS.values()) <= set(OPTIONAL_MODULES)
