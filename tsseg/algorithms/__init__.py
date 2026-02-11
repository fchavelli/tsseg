from .amoc.detector import AmocDetector
from .autoplait.detector import AutoPlaitDetector
from .clap.clap_detector import ClapDetector
from .clap.clasp_detector import ClaspDetector
from .espresso.detector import EspressoDetector
from .time2state.detector import Time2StateDetector
from .ticc.detector import TiccDetector
from .patss.detector import PatssDetector
from .hdp_hsmm.detector import HdpHsmmDetector
from .hdp_hsmm.legacy_detector import HdpHsmmLegacyDetector
from .e2usd.detector import E2USDDetector
from .binseg.detector import BinSegDetector
from .bottomup.detector import BottomUpDetector
from .eagglo.detector import EAggloDetector
from .fluss.detector import FLUSSDetector
from .ggs.detector import GreedyGaussianDetector
from .hidalgo.detector import HidalgoDetector
from .hmm.detector import HMMDetector
from .igts.detector import InformationGainDetector
from .random.detector import RandomDetector
from .bocd.detector import BOCDDetector
from .icid.detector import ICIDDetector
from .tire.detector import TireDetector
from .prophet.detector import ProphetDetector
from .pelt.detector import PeltDetector
from .dynp.detector import DynpDetector
from .kcpd.detector import KCPDDetector
from .window.detector import WindowDetector
from .tscp2.detector import TSCP2Detector
from .vqtss.detector import VQTSSDetector
from .vsax.detector import VSAXDetector
from .tglad.detector import TGLADDetector
try:
    from .tirex.detector import (  # noqa: F401
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
    _TIREX_NAMES = [
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
    _TIREX_NAMES = []

__all__ = [
    "AmocDetector",
    "AutoPlaitDetector",
    "BOCDDetector",
    "BinSegDetector",
    "BottomUpDetector",
    "ClapDetector",
    "ClaspDetector",
    "E2USDDetector",
    "EAggloDetector",
    "EspressoDetector",
    "FLUSSDetector",
    "GreedyGaussianDetector",
    "HdpHsmmDetector",
    "HdpHsmmLegacyDetector",
    "HidalgoDetector",
    #"HMMDetector",
    "ICIDDetector",
    "InformationGainDetector",
    "KCPDDetector",
    "TSCP2Detector",
    "PatssDetector",
    "RandomDetector",
    "TiccDetector",
    "Time2StateDetector",
    "TireDetector",
    "ProphetDetector",
    "PeltDetector",
    "WindowDetector",
    "DynpDetector",
    "VSAXDetector",
    "TGLADDetector",
    "VQTSSDetector",
] + _TIREX_NAMES
