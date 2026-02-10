"""Vendored subset of the `ruptures` change point detection toolkit.

This package provides the core estimators, costs, and utilities required by
`tsseg`'s change point detectors without depending on the external
`ruptures` distribution.
"""

from . import base, costs, detection, exceptions, utils  # noqa: F401

__all__ = ["base", "costs", "detection", "exceptions", "utils"]
