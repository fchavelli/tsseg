"""Re-export TiRex segmenters as detectors for the benchmark harness.

The segmenter implementations live in the ``tsseg-tirex`` sibling project
(``<project-root>/tsseg-tirex/segmentation/segmenter``), which is the
single source of truth.  We add that path at import time so that
improvements made there are picked up automatically without copying files.

The ``tirex`` model package (``tsseg-tirex/tirex/src``) is also put on
``sys.path`` so that ``tirex_extractor.py`` can ``from tirex import …``.
"""

from __future__ import annotations

import os
import sys

# ── Resolve external paths ────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root:  <project-root>/tsseg/tsseg/algorithms/tirex  →  4 levels up
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))

# 1) segmenter package + tirex_extractor.py  (tsseg-tirex/segmentation/)
_SEGMENTATION_ROOT = os.path.join(_PROJECT_ROOT, "tsseg-tirex", "segmentation")

# 2) tirex model package  (for ``from tirex import load_model``)
_TIREX_SRC = os.path.join(_SEGMENTATION_ROOT, "..", "tirex", "src")
_TIREX_SRC = os.path.normpath(_TIREX_SRC)

for _p in (_SEGMENTATION_ROOT, _TIREX_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Re-exports ────────────────────────────────────────────────────────
from segmenter import (  # noqa: E402
    TirexHiddenCPD,
    TirexCosineCPD,
    TirexL2CPD,
    TirexMMDCPD,
    TirexEnergyCPD,
    TirexDerivativeCPD,
    TirexGateRatioCPD,
    TirexForgetDropCPD,
    TirexForecastErrorCPD,
    TirexStateDetector,
    TirexStateGMM,
    TirexStateKMeans,
    TirexStateHMM,
)

__all__ = [
    "TirexHiddenCPD",
    "TirexCosineCPD",
    "TirexL2CPD",
    "TirexMMDCPD",
    "TirexEnergyCPD",
    "TirexDerivativeCPD",
    "TirexGateRatioCPD",
    "TirexForgetDropCPD",
    "TirexForecastErrorCPD",
    "TirexStateDetector",
    "TirexStateGMM",
    "TirexStateKMeans",
    "TirexStateHMM",
]
