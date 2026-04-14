from .amoc.detector import AmocDetector
from .autoplait.detector import AutoPlaitDetector
from .binseg.detector import BinSegDetector
from .bocd.detector import BOCDDetector
from .bottomup.detector import BottomUpDetector
from .changefinder.detector import ChangeFinderDetector
from .clap.clap_detector import ClapDetector
from .clap.clasp_detector import ClaspDetector
from .dynp.detector import DynpDetector
from .eagglo.detector import EAggloDetector
from .espresso.detector import EspressoDetector
from .fluss.detector import FLUSSDetector
from .ggs.detector import GreedyGaussianDetector
from .hdp_hsmm.detector import HdpHsmmDetector
from .hidalgo.detector import HidalgoDetector
from .hmm.detector import HMMDetector
from .icid.detector import ICIDDetector
from .igts.detector import InformationGainDetector
from .kcpd.detector import KCPDDetector
from .patss.detector import PatssDetector
from .pelt.detector import PeltDetector
from .random.detector import RandomDetector
from .ticc.detector import TiccDetector
from .vsax.detector import VSAXDetector
from .window.detector import WindowDetector

# --- Lazy imports for detectors with heavy optional dependencies ---
# Avoids requiring torch / tensorflow / prophet just to ``import tsseg.algorithms``
_LAZY_IMPORTS = {
    # Rbeast (vendorized C extension in c/Rbeast – build with `make`)
    "BeastDetector": (".beast.detector", "BeastDetector"),
    # PyTorch-based
    "E2USDDetector": (".e2usd.detector", "E2USDDetector"),
    "TGLADDetector": (".tglad.detector", "TGLADDetector"),
    "Time2StateDetector": (".time2state.detector", "Time2StateDetector"),
    "TireDetector": (".tire.detector", "TireDetector"),
    "VQTSSDetector": (".vqtss.detector", "VQTSSDetector"),
    # TensorFlow-based
    "TSCP2Detector": (".tscp2.detector", "TSCP2Detector"),
    "SNLDSDetector": (".snlds.detector", "SNLDSDetector"),
    # Prophet
    "ProphetDetector": (".prophet.detector", "ProphetDetector"),
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
    "BeastDetector",
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
