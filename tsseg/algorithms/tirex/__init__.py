"""TiRex-based change point detectors.

These detectors require the private ``tsseg-tirex`` repository to be
cloned alongside ``tsseg``.  When the repository is absent, this
package is importable but empty – no error is raised.
"""

try:
    from .detector import (  # noqa: F401
        TirexHiddenCPD,
        TirexCosineCPD,
        TirexL2CPD,
        TirexMMDCPD,
        TirexEnergyCPD,
        TirexDerivativeCPD,
        TirexGateRatioCPD,
        TirexForgetDropCPD,
        TirexForecastErrorCPD,
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
    ]
except ImportError:
    __all__ = []
