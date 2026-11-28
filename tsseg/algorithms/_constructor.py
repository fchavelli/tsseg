"""Compatibility for the transition to keyword-only detector constructors."""

from __future__ import annotations

import warnings
from functools import wraps


def _deprecate_positional_args(*legacy_names: str):
    """Preserve the old positional mapping while exposing the new signature.

    ``wraps`` lets inspection and estimator tooling see the keyword-only
    constructor. Keep this adapter for at least one released version; removing
    positional compatibility requires a separately approved breaking release.
    """

    def decorate(init):
        @wraps(init)
        def wrapped(self, *args, **kwargs):
            if len(args) > len(legacy_names):
                raise TypeError(
                    f"{init.__qualname__}() takes at most {len(legacy_names)} "
                    f"positional arguments ({len(args)} given)"
                )
            for name, value in zip(legacy_names, args, strict=False):
                if name in kwargs:
                    raise TypeError(
                        f"{init.__qualname__}() got multiple values for argument '{name}'"
                    )
                kwargs[name] = value
            if args:
                warnings.warn(
                    f"Passing parameters positionally to {type(self).__name__} is "
                    "deprecated; use keyword arguments. Positional compatibility "
                    "will be retained for at least one released version before "
                    "removal in a future breaking release.",
                    FutureWarning,
                    stacklevel=2,
                )
            return init(self, **kwargs)

        return wrapped

    return decorate
