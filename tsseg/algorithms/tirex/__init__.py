"""TiRex-based change point detectors."""

from .detector import (
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
