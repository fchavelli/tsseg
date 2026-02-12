"""Tag contract validation.

Every ``BaseSegmenter`` exposes metadata through *tags*.  This module ensures
that every algorithm declares the minimum required tags and that declared
values are consistent.
"""

from __future__ import annotations

import pytest

VALID_DETECTOR_TYPES = {"change_point_detection", "state_detection"}


class TestTagContract:
    """Validate the tag metadata exposed by each algorithm."""

    def test_returns_dense_present(self, algorithm):
        """``returns_dense`` tag must be declared (bool)."""
        _name, _cls, _ovr, instance = algorithm
        val = instance.get_tag("returns_dense", raise_error=False, tag_value_default=None)
        assert val is not None, f"Missing tag 'returns_dense' on {_name}"
        assert isinstance(val, bool)

    def test_detector_type_valid(self, algorithm):
        """``detector_type`` must be one of the recognised values."""
        _name, _cls, _ovr, instance = algorithm
        val = instance.get_tag("detector_type", raise_error=False, tag_value_default=None)
        assert val is not None, f"Missing tag 'detector_type' on {_name}"
        assert val in VALID_DETECTOR_TYPES, (
            f"{_name}: detector_type='{val}' not in {VALID_DETECTOR_TYPES}"
        )

    def test_at_least_one_capability(self, algorithm):
        """At least one ``capability:*`` tag must be ``True``."""
        _name, _cls, _ovr, instance = algorithm
        tags = instance.get_tags()
        caps = {k: v for k, v in tags.items() if k.startswith("capability:")}
        assert caps, f"{_name} has no capability tags at all"
        assert any(caps.values()), f"{_name}: all capability tags are False"

    def test_univariate_or_multivariate(self, algorithm):
        """At least one of univariate / multivariate must be supported."""
        _name, _cls, _ovr, instance = algorithm
        uni = instance.get_tag("capability:univariate", raise_error=False, tag_value_default=False)
        multi = instance.get_tag("capability:multivariate", raise_error=False, tag_value_default=False)
        assert uni or multi, f"{_name} supports neither univariate nor multivariate"

    def test_returns_dense_consistent_with_detector_type(self, algorithm):
        """``returns_dense`` should be consistent with ``detector_type``.

        - ``change_point_detection`` → ``returns_dense=True``
        - ``state_detection``        → ``returns_dense=False``
        """
        _name, _cls, _ovr, instance = algorithm
        dtype = instance.get_tag("detector_type", raise_error=False)
        rdense = instance.get_tag("returns_dense", raise_error=False)
        if dtype is None or rdense is None:
            pytest.skip("Tags missing")
        if dtype == "change_point_detection":
            assert rdense is True, (
                f"{_name}: detector_type='change_point_detection' but returns_dense=False"
            )
        elif dtype == "state_detection":
            assert rdense is False, (
                f"{_name}: detector_type='state_detection' but returns_dense=True"
            )

    def test_fit_is_empty_is_bool(self, algorithm):
        """``fit_is_empty`` must be a bool when present."""
        _name, _cls, _ovr, instance = algorithm
        val = instance.get_tag("fit_is_empty", raise_error=False, tag_value_default=None)
        if val is not None:
            assert isinstance(val, bool)
