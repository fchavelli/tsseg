"""The deprecated HDP-HSMM aliases are removed; ``HdpHsmmDetector`` replaces them."""

from __future__ import annotations

import pytest

import tsseg.algorithms.hdp_hsmm as hdp_hsmm
from tsseg.algorithms import HdpHsmmDetector

REMOVED_ALIASES = [
    "HdpHsmmDetectorV1",
    "HdpHsmmLegacyDetector",
    "LegacyHdpHsmmDetector",
]


@pytest.mark.parametrize("name", REMOVED_ALIASES)
def test_deprecated_alias_is_removed(name):
    with pytest.raises(AttributeError, match=name):
        getattr(hdp_hsmm, name)


def test_hdp_hsmm_package_exports_the_detector_only():
    assert hdp_hsmm.__all__ == ["HdpHsmmDetector"]
    assert hdp_hsmm.HdpHsmmDetector is HdpHsmmDetector
