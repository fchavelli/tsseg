"""Checks on the dependencies declared in ``pyproject.toml``.

Every library a detector declares in its ``python_dependencies`` tag must be
installable through an extra, and through ``tsseg[all]``; every core
dependency must be imported somewhere in the package.
"""

from __future__ import annotations

import ast
import re
import warnings
from functools import cache
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = pytest.importorskip("tomli")

import tsseg.algorithms as algorithms
from tsseg.algorithms._dependencies import EXTRAS

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

if not PYPROJECT.is_file():
    pytest.skip(
        "pyproject.toml not found (not a source checkout)", allow_module_level=True
    )

# Import name -> distribution name, when they differ beyond case, "-" and "_".
DISTRIBUTION_OF_MODULE = {"tcn": "keras-tcn"}
# Distribution name -> import name, for the core dependencies.
MODULE_OF_DISTRIBUTION = {"scikit-learn": "sklearn"}
# Libraries that are not on PyPI, hence in no extra.
NOT_ON_PYPI = {"npbad"}  # PatssDetector

_PROJECT = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
_EXTRAS = _PROJECT["optional-dependencies"]


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution(requirement: str) -> str:
    return _normalise(re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0])


def _resolve(extra: str) -> set[str]:
    """Distributions installed by ``tsseg[extra]``, nested extras expanded."""
    names = set()
    for requirement in _EXTRAS[extra]:
        nested = re.fullmatch(r"tsseg\[([^\]]*)\]", requirement.replace(" ", ""))
        if nested:
            for sub in nested.group(1).split(","):
                names |= _resolve(sub)
        else:
            names.add(_distribution(requirement))
    return names


@cache
def _imported_modules() -> frozenset[str]:
    """Top-level names of the absolute imports found under ``tsseg/``."""
    modules = set()
    for path in (ROOT / "tsseg").rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules.add(node.module.split(".")[0])
    return frozenset(modules)


@pytest.mark.parametrize("name", algorithms.__all__)
def test_python_dependencies_are_declared(name):
    try:
        detector = getattr(algorithms, name)
    except ImportError as exc:
        pytest.skip(f"{name} cannot be imported here: {exc}")
    modules = detector.get_class_tag("python_dependencies") or []
    if isinstance(modules, str):
        modules = [modules]
    core = {_distribution(r) for r in _PROJECT["dependencies"]}
    in_extras = core.union(*(_resolve(extra) for extra in _EXTRAS))
    in_all = core | _resolve("all")
    for module in set(modules) - NOT_ON_PYPI:
        dist = _normalise(DISTRIBUTION_OF_MODULE.get(module, module))
        assert dist in in_extras, f"{name} needs {module}, declared in no extra"
        assert dist in in_all, f"{name} needs {module}, not installed by tsseg[all]"


@pytest.mark.parametrize("requirement", _PROJECT["dependencies"])
def test_core_dependency_is_imported(requirement):
    dist = _distribution(requirement)
    module = MODULE_OF_DISTRIBUTION.get(dist, dist.replace("-", "_"))
    assert module in _imported_modules(), f"{requirement} is imported nowhere in tsseg"


@pytest.mark.parametrize("extra", sorted(EXTRAS))
def test_error_message_extras_install_their_modules(extra):
    """The extras named by the missing-dependency error install those modules."""
    assert extra in _EXTRAS, f"tsseg[{extra}] is not an extra"
    installed = _resolve(extra)
    for module in EXTRAS[extra]:
        dist = _normalise(DISTRIBUTION_OF_MODULE.get(module, module))
        assert dist in installed, f"tsseg[{extra}] does not install {module}"


@pytest.mark.parametrize("name", algorithms.__all__)
def test_error_message_names_an_extra(name):
    """Every optional module of a detector is installed by an extra of EXTRAS."""
    modules = getattr(algorithms, name).get_class_tag("python_dependencies") or []
    covered = set().union(*EXTRAS.values())
    for module in set(modules) - NOT_ON_PYPI:
        assert module in covered, f"{name}: add the extra of {module} to EXTRAS"
