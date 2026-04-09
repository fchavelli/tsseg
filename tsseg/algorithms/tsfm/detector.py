"""Re-export tsseg-fm detectors as part of the tsseg algorithms namespace.

If ``tsseg_fm`` is pip-installed, all FM-based detectors are available via::

    from tsseg.algorithms.tsfm.detector import FMCosimCPD, FMStateGMM, ...

If ``tsseg_fm`` is not installed, this module exports nothing and emits
a warning.
"""

from __future__ import annotations

try:
    from tsseg_fm import (  # noqa: F401
        FML2CPD,
        FMMMDCPD,
        FMAgglomCPD,
        FMBinSegCPD,
        FMBottomUpCPD,
        FMBottomUpStabCPD,
        FMClaSPCPD,
        FMCosimCPD,
        FMDerivativeCPD,
        FMDistProfileCPD,
        FMDynpCPD,
        FMEnergyCPD,
        FMHiddenCPD,
        FMKernelCPD,
        FMStateDetector,
        FMStateGMM,
        FMStateHMM,
        FMStateKMeans,
    )

    __all__ = [
        "FMHiddenCPD",
        "FMCosimCPD",
        "FML2CPD",
        "FMMMDCPD",
        "FMEnergyCPD",
        "FMDerivativeCPD",
        "FMStateDetector",
        "FMStateGMM",
        "FMStateKMeans",
        "FMStateHMM",
        "FMBottomUpCPD",
        "FMBottomUpStabCPD",
        "FMDistProfileCPD",
        "FMKernelCPD",
        "FMBinSegCPD",
        "FMDynpCPD",
        "FMAgglomCPD",
        "FMClaSPCPD",
    ]
except ImportError:
    import warnings as _warnings

    _warnings.warn(
        "tsseg-fm is not installed — FM-based detectors are unavailable. "
        "Install with:  pip install -e ../tsseg-fm",
        ImportWarning,
        stacklevel=2,
    )
    __all__ = []
