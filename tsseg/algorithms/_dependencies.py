"""Check the optional dependencies a detector declares.

A detector lists the modules it needs in its ``python_dependencies`` tag. Its
module imports without them, so that ``from tsseg.algorithms import *`` works
on a bare ``pip install tsseg``; the check happens at construction instead
(``BaseSegmenter.__init__``), as in aeon, and names the extra to install.
"""

from __future__ import annotations

from collections.abc import Iterable
from importlib.util import find_spec

__all__ = ["EXTRAS", "check_python_dependencies", "missing_dependencies"]

#: Extras of ``pyproject.toml`` -> top-level modules they install (only the
#: modules that a detector declares; ``tests/test_dependencies.py`` checks that
#: each extra installs them).
EXTRAS: dict[str, frozenset[str]] = {
    "accelerators": frozenset({"numba"}),
    "aeon": frozenset({"aeon"}),
    "beast": frozenset({"Rbeast"}),
    "changeforest": frozenset({"changeforest"}),
    "fluss": frozenset({"stumpy"}),
    "prophet": frozenset({"prophet"}),
    "snlds": frozenset({"tensorflow", "tensorflow_probability", "tf_keras"}),
    "tglad": frozenset({"torch", "networkx"}),
    "torch": frozenset({"torch"}),
    "tscp2": frozenset({"tensorflow", "tcn"}),
}


def missing_dependencies(modules: Iterable[str]) -> list[str]:
    """Return the modules of ``modules`` that cannot be found, in order.

    The modules are looked up, not imported.
    """
    missing = []
    for module in modules:
        try:
            found = find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(module)
    return missing


def _extras_for(missing: list[str], required: set[str]) -> tuple[list[str], set[str]]:
    """Extras that install ``missing`` (greedy cover), and the modules left over.

    Extras that install only modules of ``required`` come first, so that a
    module shared by two extras (``tensorflow``) leads to the detector's own;
    among those, the one covering most missing modules, then the smallest.
    """
    preferred = [e for e, mods in EXTRAS.items() if mods <= required]
    others = [e for e in EXTRAS if e not in preferred]
    uncovered = set(missing)
    chosen: list[str] = []
    for pool in (preferred, others):
        while uncovered:
            gains = {e: len(EXTRAS[e] & uncovered) for e in pool if e not in chosen}
            best = max(
                sorted(gains), key=lambda e: (gains[e], -len(EXTRAS[e])), default=None
            )
            if best is None or not gains[best]:
                break
            chosen.append(best)
            uncovered -= EXTRAS[best]
    return sorted(chosen), uncovered


def check_python_dependencies(estimator) -> None:
    """Raise ``ModuleNotFoundError`` if a declared dependency is missing.

    Parameters
    ----------
    estimator : BaseSegmenter instance or class
        Its ``python_dependencies`` class tag (``None`` or a list of top-level
        module names) is checked.

    Raises
    ------
    ModuleNotFoundError
        If a module cannot be found. The message names the missing modules and
        the extras of tsseg that install them, e.g. ``ClapDetector requires
        numba, aeon: pip install "tsseg[accelerators,aeon]"``.
    """
    cls = estimator if isinstance(estimator, type) else type(estimator)
    modules = cls.get_class_tag(
        "python_dependencies", tag_value_default=None, raise_error=False
    )
    if isinstance(modules, str):
        modules = [modules]
    missing = missing_dependencies(modules or [])
    if not missing:
        return
    extras, uncovered = _extras_for(missing, set(modules))
    message = f"{cls.__name__} requires {', '.join(missing)}"
    if extras and not uncovered:
        message += f': pip install "tsseg[{",".join(extras)}]"'
    elif len(missing) == 1:
        message += ", which is not installed."
    else:
        message += ", which are not installed."
    raise ModuleNotFoundError(message, name=missing[0])
