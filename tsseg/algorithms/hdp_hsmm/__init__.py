from .detector import HdpHsmmDetector

__all__ = ["HdpHsmmDetector"]


def __getattr__(name: str):
    if name == "HdpHsmmLegacyDetector":
        import warnings
        warnings.warn(
            "HdpHsmmLegacyDetector is deprecated and will be removed in a "
            "future release. Use HdpHsmmDetector instead (pure NumPy/SciPy, "
            "no pyhsmm dependency required).",
            FutureWarning,
            stacklevel=2,
        )
        from .legacy_detector import HdpHsmmLegacyDetector
        return HdpHsmmLegacyDetector
    if name == "LegacyHdpHsmmDetector":
        import warnings
        warnings.warn(
            "LegacyHdpHsmmDetector is deprecated and will be removed in a "
            "future release. Use HdpHsmmDetector instead.",
            FutureWarning,
            stacklevel=2,
        )
        from .legacy_pyhsmm import LegacyHdpHsmmDetector
        return LegacyHdpHsmmDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
