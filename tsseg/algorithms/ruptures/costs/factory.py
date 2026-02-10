"""Factory function returning cost objects by model name."""

from __future__ import annotations

from typing import Any

from ..base import BaseCost


def cost_factory(model: str, *args: Any, **kwargs: Any) -> BaseCost:
    """Return a cost class registered under ``model``."""

    for subclass in BaseCost.__subclasses__():
        if subclass.model == model:
            return subclass(*args, **kwargs)
    raise ValueError(f"Unknown cost model: {model}")
