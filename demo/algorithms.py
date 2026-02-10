"""Utilities to expose available detectors with metadata for the demo UI."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from tsseg import algorithms as algorithms_pkg

DetectorType = str


@dataclass
class DetectorInfo:
    name: str
    cls: type
    detector_type: DetectorType
    tags: dict[str, Any]


def _iter_exported_detectors() -> Iterable[Tuple[str, type]]:
    exported = getattr(algorithms_pkg, "__all__", [])
    for name in exported:
        try:
            obj = getattr(algorithms_pkg, name)
        except AttributeError:
            continue
        if inspect.isclass(obj) and hasattr(obj, "_tags"):
            yield name, obj


def get_detectors_by_type(detector_type: DetectorType) -> Dict[str, DetectorInfo]:
    """Return detectors grouped by detector_type tag.

    Parameters
    ----------
    detector_type:
        The detector type to filter on (e.g. "change_point_detection", "state_detection").
    """

    results: Dict[str, DetectorInfo] = {}
    for name, cls in _iter_exported_detectors():
        tags = getattr(cls, "_tags", {}) or {}
        if tags.get("detector_type") != detector_type:
            continue
        results[name] = DetectorInfo(name=name, cls=cls, detector_type=detector_type, tags=tags)
    return results


def get_constructor_signature(cls: type) -> inspect.Signature:
    """Return the constructor signature excluding ``self``."""

    sig = inspect.signature(cls.__init__)
    params = [
        param
        for param in sig.parameters.values()
        if param.name != "self" and param.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    ]
    return inspect.Signature(params)


def instantiate_detector(cls: type, params: Dict[str, Any]):
    """Instantiate a detector with validated parameters."""

    try:
        return cls(**params)
    except TypeError as exc:  # pragma: no cover - surface error to UI
        raise ValueError(f"Failed to instantiate {cls.__name__}: {exc}") from exc
