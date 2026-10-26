"""
tsseg: A Python library for time series segmentation, compatible with aeon.

``import tsseg`` stays light: the detectors of :mod:`tsseg.algorithms` are
imported on first access to ``tsseg.algorithms``.
"""

from .data.datasets import load_mocap

__version__ = "0.3.0"

__all__ = ["load_mocap"]

#: Subpackages imported on first attribute access.
_LAZY_SUBMODULES = ("algorithms",)


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib

        # Importing the subpackage also binds it as an attribute of tsseg.
        return importlib.import_module(f".{name}", __name__)
    if name == "TGLADDetector":
        import warnings

        warnings.warn(
            "Importing TGLADDetector from the top-level tsseg package is "
            "deprecated and will be removed in a future release; use "
            "`from tsseg.algorithms import TGLADDetector` instead.",
            FutureWarning,
            stacklevel=2,
        )
        from .algorithms import TGLADDetector

        return TGLADDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
