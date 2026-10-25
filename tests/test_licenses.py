"""License notices of the code that tsseg adapts from aeon.

aeon is distributed under the BSD 3-Clause License, whose first condition asks
redistributions of source code to retain its copyright notice, conditions and
disclaimer. The detectors adapted from aeon keep a copy of the license in their
directory; ``tsseg/algorithms/base.py``, which sits next to every detector
directory, keeps it in its opening comment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ALGORITHMS = Path(__file__).resolve().parents[1] / "tsseg" / "algorithms"
AEON_HOLDER = "Copyright (c) The aeon developers."

AEON_COPIES = sorted(
    path
    for path in ALGORITHMS.glob("*/LICENSE")
    if AEON_HOLDER in path.read_text(encoding="utf-8")
)


def _license_text(path: Path) -> list[str]:
    """Lines of a license copy, from its title on (the preamble is dropped)."""
    text = path.read_text(encoding="utf-8")
    return text[text.index("BSD 3-Clause License") :].strip().splitlines()


def _opening_comment(path: Path) -> list[str]:
    """Comment lines that open a module, without their ``#`` prefix."""
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line[2:] if line.startswith("# ") else line[1:])
    return lines


def _contains_block(lines: list[str], block: list[str]) -> bool:
    """Whether ``block`` appears in ``lines`` as consecutive lines."""
    size = len(block)
    return any(
        lines[start : start + size] == block for start in range(len(lines) - size + 1)
    )


def test_aeon_copies_are_found():
    """The detectors adapted from aeon ship a copy of its license."""
    assert {path.parent.name for path in AEON_COPIES} >= {"eagglo", "hidalgo", "igts"}


@pytest.mark.parametrize("copy", AEON_COPIES, ids=lambda path: path.parent.name)
def test_base_module_keeps_the_aeon_notice(copy):
    """``base.py`` opens with aeon's license, word for word as in each copy."""
    header = _opening_comment(ALGORITHMS / "base.py")
    assert _contains_block(header, _license_text(copy))
