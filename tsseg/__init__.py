"""
tsseg: A Python library for time series segmentation, compatible with aeon.
"""

# Minimal fallback patch for edge cases where .pth file doesn't work
try:
    import numpy as np
    import sys
    import types
    # Only patch if not already done by .pth file
    if 'numpy.core.umath_tests' not in sys.modules:
        umath_tests = types.ModuleType('numpy.core.umath_tests')
        umath_tests.inner1d = lambda x, y: np.einsum('ij,ij->i', x, y)
        sys.modules['numpy.core.umath_tests'] = umath_tests
except ImportError:
    pass

__version__ = "0.1.0"

from .data.datasets import load_mocap

try:
    from .algorithms.tglad.detector import TGLADDetector
except ImportError:
    TGLADDetector = None

__all__ = [
    "load_mocap",
    "TGLADDetector",
]
