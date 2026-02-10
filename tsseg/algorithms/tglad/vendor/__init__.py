"""Vendorized uGLAD package used by the TGLAD detector.

This module registers the vendored implementation under the top-level
``uGLAD`` package name so that existing absolute imports continue to work
without modification.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _register_submodules(pkg: ModuleType, package_name: str) -> None:
    """Ensure submodules are visible through ``sys.modules``."""

    for sub_name in pkg.__all__:
        full_name = f"{package_name}.{sub_name}"
        if full_name in sys.modules:
            continue
        module = importlib.import_module(f".{sub_name}", package=package_name)
        sys.modules[full_name] = module
        if hasattr(module, "__all__"):
            # Recursively register nested packages
            _register_submodules(module, full_name)


def ensure_vendor_imports() -> None:
    """Install the vendored package into ``sys.modules`` if needed."""

    if "uGLAD" in sys.modules:
        return

    from . import uGLAD as vendored_pkg

    sys.modules["uGLAD"] = vendored_pkg
    _register_submodules(vendored_pkg, vendored_pkg.__name__)


__all__ = ["ensure_vendor_imports"]
