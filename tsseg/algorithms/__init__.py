from .amoc.detector import AmocDetector
from .autoplait.detector import AutoPlaitDetector
from .changefinder.detector import ChangeFinderDetector
from .clap.clap_detector import ClapDetector
from .clap.clasp_detector import ClaspDetector
from .espresso.detector import EspressoDetector
from .time2state.detector import Time2StateDetector
from .ticc.detector import TiccDetector
from .patss.detector import PatssDetector
from .hdp_hsmm.detector import HdpHsmmDetector
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
from .vqtss.detector import VQTSSDetector
from .vsax.detector import VSAXDetector
from .tglad.detector import TGLADDetector


# --- Lazy imports for TensorFlow-based detectors to avoid loading TF on ``import tsseg`` ---
_LAZY_IMPORTS = {
    "TSCP2Detector": (".tscp2.detector", "TSCP2Detector"),
    "SNLDSDetector": (".snlds.detector", "SNLDSDetector"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr)
        globals()[name] = value  # cache for subsequent accesses
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AmocDetector",
    "AutoPlaitDetector",
    "BinSegDetector",
    "BOCDDetector",
    "BottomUpDetector",
    "ChangeFinderDetector",
    "ClapDetector",
    "ClaspDetector",
    "DynpDetector",
    "E2USDDetector",
    "EAggloDetector",
    "EspressoDetector",
    "FLUSSDetector",
    "GreedyGaussianDetector",
    "HdpHsmmDetector",
    "HidalgoDetector",
    "HMMDetector",
    "ICIDDetector",
    "InformationGainDetector",
    "KCPDDetector",
    "PatssDetector",
    "PeltDetector",
    "ProphetDetector",
    "RandomDetector",
    "SNLDSDetector",
    "TGLADDetector",
    "TiccDetector",
    "Time2StateDetector",
    "TireDetector",
    "TSCP2Detector",
    "VQTSSDetector",
    "VSAXDetector",
    "WindowDetector",
]
