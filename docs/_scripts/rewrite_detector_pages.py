"""Rewrite the per-detector RST pages into a clean, short-titled form.

Reads ``docs/api/tsseg.algorithms.<name>.rst`` and writes
``docs/detectors/<name>.rst`` with:
- the title replaced by the human-friendly display name (e.g. ``BinSeg``),
- the ``Subpackages`` block removed,
- the ``Submodules`` / ``X.detector module`` headers stripped (the
  ``.. automodule::`` directive itself is preserved under a single
  ``API reference`` section),
- the trailing ``Module contents`` block removed.

Run from the repo root:

    python docs/_scripts/rewrite_detector_pages.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "api"
DST = ROOT / "detectors"

# Display name per package short name (file stem after "tsseg.algorithms.").
DISPLAY: dict[str, str] = {
    "amoc": "AMOC",
    "autoplait": "AutoPlait",
    "beast": "BEAST",
    "binseg": "BinSeg",
    "bocd": "BOCD",
    "bottomup": "BottomUp",
    "changefinder": "ChangeFinder",
    "clap": "CLaP / ClaSP",
    "dynp": "DynP",
    "e2usd": "E2USD",
    "eagglo": "E-Agglo",
    "espresso": "ESPRESSO",
    "fluss": "FLUSS",
    "ggs": "GreedyGaussian (GGS)",
    "hdp_hsmm": "HDP-HSMM",
    "hidalgo": "Hidalgo",
    "hmm": "HMM",
    "icid": "iCID",
    "igts": "InformationGain (IGTS)",
    "kcpd": "KCPD",
    "patss": "PaTSS",
    "pelt": "PELT",
    "prophet": "Prophet",
    "random": "Random (baseline)",
    "ruptures": "Ruptures (vendored)",
    "tglad": "tGLAD",
    "ticc": "TICC",
    "time2state": "Time2State",
    "tire": "TIRE",
    "tscp2": "TS-CP2",
    "vqtss": "VQ-TSS",
    "vsax": "vSAX",
    "window": "Window",
}

# Skip these (PaTSS internal subpackages -- not user-facing).
SKIP_STEMS = {
    "patss.algorithms",
    "patss.embedding",
    "patss.segmentation",
}

FILE_RE = re.compile(r"^tsseg\.algorithms\.(?P<stem>[a-z0-9_.]+)\.rst$")


def rewrite_title(text: str, display: str) -> str:
    """Replace the first 2 lines (title + underline) with ``display``."""
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    # Drop the original title and its underline.
    body = "\n".join(lines[2:]).lstrip("\n")
    underline = "=" * len(display)
    return f"{display}\n{underline}\n\n{body}"


def strip_subpackages(text: str) -> str:
    """Remove ``Subpackages`` section and its toctree."""
    return re.sub(
        r"\nSubpackages\n-+\n.*?(?=\n[A-Z][^\n]*\n-+\n|\Z)",
        "\n",
        text,
        flags=re.DOTALL,
    )


def strip_submodules_headers(text: str) -> str:
    """Drop the ``Submodules`` header and per-module sub-headers.

    Replaces the whole ``Submodules ...`` block (up to ``Module contents`` or
    end of file) by a single ``API reference`` section that keeps every
    ``.. automodule::`` directive that was inside.
    """
    m = re.search(r"\nSubmodules\n-+\n", text)
    if not m:
        return text
    head, tail = text[: m.start()], text[m.end() :]
    # End of the Submodules block: either ``Module contents`` header or EOF.
    end_m = re.search(r"\nModule contents\n-+\n.*\Z", tail, flags=re.DOTALL)
    if end_m:
        block = tail[: end_m.start()]
    else:
        block = tail
    # Collect all automodule blocks (directive + options).
    autos = re.findall(
        r"\.\. automodule:: [^\n]+\n(?:   :[^\n]+\n)*",
        block,
    )
    if not autos:
        return head.rstrip() + "\n"
    api = "\n\nAPI reference\n-------------\n\n" + "\n".join(autos)
    return head.rstrip() + api


def strip_module_contents(text: str) -> str:
    """Drop the trailing ``Module contents`` block."""
    return re.sub(
        r"\nModule contents\n-+\n.*\Z",
        "\n",
        text,
        flags=re.DOTALL,
    )


def process(path: Path) -> tuple[str, str] | None:
    m = FILE_RE.match(path.name)
    if not m:
        return None
    stem = m.group("stem")
    if stem in SKIP_STEMS:
        return None
    display = DISPLAY.get(stem)
    if display is None:
        return None
    text = path.read_text(encoding="utf-8")
    text = strip_subpackages(text)
    text = strip_submodules_headers(text)
    text = strip_module_contents(text)
    text = rewrite_title(text, display)
    return stem, text.rstrip() + "\n"


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    produced: list[tuple[str, str]] = []
    for p in sorted(SRC.glob("tsseg.algorithms.*.rst")):
        result = process(p)
        if result is None:
            continue
        stem, text = result
        out = DST / f"{stem}.rst"
        out.write_text(text, encoding="utf-8")
        produced.append((stem, DISPLAY[stem]))
    print(f"Wrote {len(produced)} detector pages to {DST}")
    for stem, display in produced:
        print(f"  {stem:<14} -> {display}")


if __name__ == "__main__":
    main()
