"""Typographic rules of the documentation sources."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EM_DASH = "—"


def _documentation_files() -> list[Path]:
    """README, citation, Sphinx sources and assets, package READMEs."""
    docs = ROOT / "docs"
    files = [ROOT / "README.md", ROOT / "CITATION.cff"]
    for pattern in ("*.rst", "*.md", "*.py", "*.html", "*.js", "*.css", "*.svg"):
        files += [p for p in docs.rglob(pattern) if "_build" not in p.parts]
    # vendored code keeps the punctuation of its upstream
    files += [
        p
        for p in (ROOT / "tsseg").rglob("README.md")
        if not {"ruptures", "vendor"} & set(p.parts)
    ]
    return sorted(p for p in files if p.is_file())


def test_documentation_files_are_found():
    names = {p.relative_to(ROOT).as_posix() for p in _documentation_files()}
    assert {"README.md", "docs/index.rst", "docs/_static/js/taxonomy.js"} <= names


def test_no_em_dash_in_documentation():
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in _documentation_files()
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if EM_DASH in line
    ]
    assert not offenders, (
        "Em dashes are not used in the documentation: write an en dash (–), a "
        "colon or a new sentence instead.\n" + "\n".join(offenders)
    )
