"""Every module of the package compiles without a warning.

Python only compiles a module the first time it is imported (afterwards it
reads the cached bytecode), so a warning issued at compile time, such as an
invalid escape sequence in a docstring, reaches users once per installation and
is easy to miss in a local run. Invalid escape sequences are a
``DeprecationWarning`` up to Python 3.11, a ``SyntaxWarning`` from 3.12, and
will become a ``SyntaxError`` in a future version. This test compiles every
``.py`` file of the imported package, vendored code included, and fails on any
warning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import tsseg

PACKAGE_DIR = Path(tsseg.__file__).resolve().parent


def _compile_warnings(path: Path) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    return [
        f"{path.relative_to(PACKAGE_DIR.parent)}:{w.lineno}: "
        f"{w.category.__name__}: {w.message}"
        for w in caught
    ]


def test_package_sources_compile_without_warnings():
    sources = sorted(PACKAGE_DIR.rglob("*.py"))
    assert sources, f"no Python file found under {PACKAGE_DIR}"
    found = [message for path in sources for message in _compile_warnings(path)]
    assert not found, "compile-time warnings:\n" + "\n".join(found)
