from __future__ import annotations

import pytest


def test_algorithm_tags_presence(algorithm_instance):
    instance, _ = algorithm_instance
    required_keys = {"returns_dense", "detector_type"}
    for key in required_keys:
        value = instance.get_tag(key)
        assert value is not None, f"Missing tag '{key}' for {instance.__class__.__name__}"


def test_algorithm_capability_tags(algorithm_instance):
    instance, _ = algorithm_instance
    capability_keys = [key for key in instance._tags if key.startswith("capability:")]
    assert capability_keys, f"No capability tags defined for {instance.__class__.__name__}"
