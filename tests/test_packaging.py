"""Checks on the package data that ``pyproject.toml`` puts in the wheel.

setuptools ships the non-Python files of ``tsseg`` that the
``[tool.setuptools.package-data]`` globs select, expanded from the ``tsseg``
directory (``glob(..., recursive=True)``, as ``build_py`` does), minus those
that match an ``[tool.setuptools.exclude-package-data]`` pattern
(``fnmatch``). These tests expand the same patterns, so that a vendored
component cannot lose its license in the wheel and MATLAB files cannot enter
it.
"""

from __future__ import annotations

import fnmatch
import glob
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

import tsseg

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_DIR = Path(tsseg.__file__).resolve().parent

if not PYPROJECT.is_file():
    pytest.skip(
        "pyproject.toml not found (not a source checkout)", allow_module_level=True
    )
if PACKAGE_DIR.parent != ROOT:
    pytest.skip(
        "tsseg is not imported from this source checkout", allow_module_level=True
    )

_PROJECT_CONFIG = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
_SETUPTOOLS = _PROJECT_CONFIG["tool"]["setuptools"]
INCLUDED = _SETUPTOOLS["package-data"]["tsseg"]
EXCLUDED = [
    pattern
    for patterns in _SETUPTOOLS.get("exclude-package-data", {}).values()
    for pattern in patterns
]
LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING)(\.[A-Za-z]+)?$", re.IGNORECASE)


def _shipped(package_dir: Path) -> set[str]:
    """Paths, relative to ``package_dir``, of the data files put in the wheel."""
    selected = {
        Path(match).as_posix()
        for pattern in INCLUDED
        for match in glob.glob(pattern, root_dir=package_dir, recursive=True)
        if (package_dir / match).is_file()
    }
    return {
        path
        for path in selected
        if not any(fnmatch.fnmatch(path, pattern) for pattern in EXCLUDED)
    }


def test_every_license_file_is_shipped():
    licenses = {
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file() and LICENSE_NAME.match(path.name)
    }
    assert licenses, "no license file found under tsseg/"
    missing = sorted(licenses - _shipped(PACKAGE_DIR))
    assert not missing, f"license files left out of the wheel: {missing}"


@pytest.mark.parametrize("name", ["data.mat", "script.m", "live.mlx"])
def test_matlab_files_are_not_shipped(tmp_path, name):
    """A MATLAB file left next to the code stays out of the wheel."""
    nested = tmp_path / "algorithms" / "foo"
    nested.mkdir(parents=True)
    (nested / name).write_text("", encoding="utf-8")
    (nested / "LICENSE").write_text("", encoding="utf-8")
    assert _shipped(tmp_path) == {"algorithms/foo/LICENSE"}
