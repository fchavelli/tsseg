"""License notices of the third-party code that tsseg adapts.

aeon is distributed under the BSD 3-Clause License, whose first condition asks
redistributions of source code to retain its copyright notice, conditions and
disclaimer. The detectors adapted from aeon keep a copy of the license in their
directory; ``tsseg/algorithms/base.py``, which sits next to every detector
directory, keeps it in its opening comment.

Some detector modules also copy code from a project other than the one the
detector comes from. The Apache License 2.0 asks redistributions to give a copy
of the license and to keep the attribution notices of the ``NOTICE`` file; the
MIT License asks them to keep the copyright notice and the permission notice.
The ``LICENSE`` file of these detector directories carries them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ALGORITHMS = Path(__file__).resolve().parents[1] / "tsseg" / "algorithms"
AEON_HOLDER = "Copyright (c) The aeon developers."

# Full text of the Apache License 2.0, as shipped with the vendored SNLDS code.
APACHE_COPY = ALGORITHMS / "snlds" / "vendor" / "LICENSE"
FRANCESCHI_NOTICE = (
    "Copyright 2019 Jean-Yves Franceschi, Aymeric Dieuleveut, Martin Jaggi"
)
MIT_PERMISSION = (
    "The above copyright notice and this permission notice shall be included in all"
)

# Detector directories whose module holds code taken from another project:
# the copyright line of the Apache ``NOTICE`` and the MIT copyright lines that
# the ``LICENSE`` of the directory must keep.
BORROWED_CODE = {
    # Causal CNN encoder and Dataset of Franceschi et al. (2019); Chomp1d and
    # CausalConvolutionBlock partly from locuslab/TCN.
    "time2state": (
        FRANCESCHI_NOTICE,
        ["Copyright (c) 2022 Chengyu Wang", "Copyright (c) 2018 CMU Locus Lab"],
    ),
    # Dataset of Franceschi et al. (2019); moving_avg and series_decomp from
    # thuml/Autoformer. The rest of the directory has no published license.
    "e2usd": (FRANCESCHI_NOTICE, ["Copyright (c) 2021 THUML @ Tsinghua University"]),
}

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


def _lines(text: str) -> list[str]:
    """Lines of ``text`` without trailing whitespace."""
    return [line.rstrip() for line in text.splitlines()]


def _borrowed_license(directory: str) -> str:
    path = ALGORITHMS / directory / "LICENSE"
    assert path.is_file(), f"{directory}/ has no LICENSE file"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("directory", sorted(BORROWED_CODE))
def test_borrowed_apache_code_ships_the_license(directory):
    """The full Apache License 2.0 text is in the directory's ``LICENSE``."""
    apache = _lines(APACHE_COPY.read_text(encoding="utf-8").strip("\n"))
    assert _contains_block(_lines(_borrowed_license(directory)), apache)


@pytest.mark.parametrize("directory", sorted(BORROWED_CODE))
def test_borrowed_apache_code_keeps_the_notice(directory):
    """The attribution of the upstream ``NOTICE`` file is kept."""
    notice, _ = BORROWED_CODE[directory]
    assert notice in _borrowed_license(directory)


@pytest.mark.parametrize("directory", sorted(BORROWED_CODE))
def test_borrowed_mit_code_keeps_its_notices(directory):
    """Each MIT copyright line comes with its own permission notice."""
    _, holders = BORROWED_CODE[directory]
    text = _borrowed_license(directory)
    for holder in holders:
        assert holder in text
        assert MIT_PERMISSION in text.split(holder, 1)[1]
    assert text.count(MIT_PERMISSION) == len(holders)
