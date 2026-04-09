"""Declarative parameter constraint system for tsseg detectors.

This module provides a lightweight, pure-Python framework for declaring,
introspecting and validating the hyper-parameters of segmentation algorithms.
Each detector can declare a ``_parameter_schema`` class attribute that maps
parameter names to :class:`ParamDef` descriptors.  A special key
``"_cross_constraints"`` holds a list of multi-parameter or data-dependent
constraints.

The system is intentionally **opt-in**: detectors without a schema continue
to work exactly as before.  The schema is consumed by:

* :func:`validate_params` — called automatically in ``BaseSegmenter.fit()``
  to raise clear errors early.
* :func:`get_parameter_schema` / :func:`get_ui_hints` — used by the demo
  application (or any frontend) to render intelligent controls.

Design decisions
----------------
* Pure Python objects (not YAML/JSON) — type-safe, testable, IDE-friendly.
* Follows the same MRO-based inheritance pattern as ``_tags``.
* Inspired by scikit-learn's ``_parameter_constraints`` (sklearn ≥ 1.2) but
  tailored to tsseg's needs (data-dependent bounds, mutual exclusion, groups).
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

__all__ = [
    # Core constraint objects
    "Interval",
    "StrOptions",
    "Options",
    "HasType",
    "MutuallyExclusive",
    "ConditionalRequired",
    "DependsOn",
    "DataDependent",
    # Parameter descriptor
    "ParamDef",
    # Introspection / validation public API
    "get_parameter_schema",
    "validate_params",
    "get_ui_hints",
    "CROSS_CONSTRAINTS_KEY",
]

# Key used inside ``_parameter_schema`` dicts for cross-parameter constraints.
CROSS_CONSTRAINTS_KEY = "_cross_constraints"

# ---------------------------------------------------------------------------
# Closed-side helper
# ---------------------------------------------------------------------------


class Closed(Enum):
    """Which side(s) of an :class:`Interval` are closed (inclusive)."""

    LEFT = auto()
    RIGHT = auto()
    BOTH = auto()
    NEITHER = auto()


# ---------------------------------------------------------------------------
# Constraint classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """Numeric interval constraint.

    Parameters
    ----------
    type : type
        Expected numeric type — typically ``int`` or ``float``.
    low : float | int | None
        Lower bound (``None`` = unbounded).
    high : float | int | None
        Upper bound (``None`` = unbounded).
    closed : Closed
        Which side(s) are inclusive.
    """

    type: type
    low: float | int | None
    high: float | int | None
    closed: Closed = Closed.BOTH

    # -- public API --------------------------------------------------------

    def validate(self, value: Any, name: str = "value") -> str | None:
        """Return an error message or ``None`` if *value* is valid."""
        if value is None:
            return None  # ``None`` acceptance is handled by ``Options``
        if not isinstance(value, (int, float)):
            return f"{name} must be numeric, got {type(value).__name__}"

        # Check type (allow int for float intervals)
        if self.type is int and not isinstance(value, int):
            return f"{name} must be an integer, got {type(value).__name__}"

        if self.low is not None:
            if self.closed in (Closed.LEFT, Closed.BOTH):
                if value < self.low:
                    return f"{name}={value} must be >= {self.low}"
            else:
                if value <= self.low:
                    return f"{name}={value} must be > {self.low}"

        if self.high is not None:
            if self.closed in (Closed.RIGHT, Closed.BOTH):
                if value > self.high:
                    return f"{name}={value} must be <= {self.high}"
            else:
                if value >= self.high:
                    return f"{name}={value} must be < {self.high}"

        return None

    def effective_bounds(
        self, data_ctx: dict[str, Any] | None = None
    ) -> tuple[float | int | None, float | int | None]:
        """Return ``(low, high)`` with any data-context placeholders resolved."""
        return (self.low, self.high)

    # -- display helpers ---------------------------------------------------

    def __str__(self) -> str:
        lo = "-∞" if self.low is None else str(self.low)
        hi = "+∞" if self.high is None else str(self.high)
        lb = "[" if self.closed in (Closed.LEFT, Closed.BOTH) else "("
        rb = "]" if self.closed in (Closed.RIGHT, Closed.BOTH) else ")"
        return f"{lb}{lo}, {hi}{rb}"


@dataclass(frozen=True)
class StrOptions:
    """Constraint: value must be one of the given strings.

    Parameters
    ----------
    options : set[str]
        Allowed string values.
    """

    options: frozenset[str]

    def __init__(self, options: set[str] | frozenset[str]):
        object.__setattr__(self, "options", frozenset(options))

    def validate(self, value: Any, name: str = "value") -> str | None:
        if value is None:
            return None
        if value not in self.options:
            return f"{name}={value!r} must be one of {sorted(self.options)}"
        return None

    def __str__(self) -> str:
        return "{" + ", ".join(sorted(self.options)) + "}"


@dataclass(frozen=True)
class Options:
    """Typed constraint that additionally accepts specific sentinel values.

    Typically used to express ``int | None`` or ``str | int``.

    Parameters
    ----------
    type : type | tuple[type, ...]
        Expected primary type(s).
    sentinels : frozenset
        Additional acceptable values (e.g. ``{None}``).
    """

    type: type | tuple[type, ...]
    sentinels: frozenset

    def __init__(self, type: type | tuple[type, ...], sentinels: set | frozenset):
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "sentinels", frozenset(sentinels))

    def validate(self, value: Any, name: str = "value") -> str | None:
        if value in self.sentinels:
            return None
        if not isinstance(value, self.type):
            return (
                f"{name} must be of type {self.type} or one of {self.sentinels}, "
                f"got {type(value).__name__}"
            )
        return None


@dataclass(frozen=True)
class HasType:
    """Constraint: value must be an instance of one of the given types.

    Parameters
    ----------
    types : tuple[type, ...]
        Acceptable types.
    """

    types: tuple[type, ...]

    def validate(self, value: Any, name: str = "value") -> str | None:
        if value is None:
            return None
        if not isinstance(value, self.types):
            return f"{name} must be an instance of {self.types}, got {type(value).__name__}"
        return None


# ---------------------------------------------------------------------------
# Cross-parameter and data-dependent constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutuallyExclusive:
    """Exactly *required_count* parameters among *params* must be non-None.

    Parameters
    ----------
    params : tuple[str, ...]
        Names of the mutually-exclusive parameters.
    required_count : int
        How many of them must be set (non-None).  Typically ``1``.
    """

    params: tuple[str, ...]
    required_count: int = 1

    def __init__(self, params: Sequence[str], required_count: int = 1):
        object.__setattr__(self, "params", tuple(params))
        object.__setattr__(self, "required_count", required_count)

    def validate(self, param_values: dict[str, Any], name: str = "") -> str | None:
        count = sum(1 for p in self.params if param_values.get(p) is not None)
        if count != self.required_count:
            return (
                f"Exactly {self.required_count} of {list(self.params)} must be set "
                f"(non-None), but {count} are set"
            )
        return None


@dataclass(frozen=True)
class ConditionalRequired:
    """A parameter is required when a condition on other params is true.

    Parameters
    ----------
    param : str
        The parameter that is conditionally required.
    condition : str
        A simple expression evaluated against the parameter dict, e.g.
        ``"semi_supervised == True"``.
    """

    param: str
    condition: str

    def validate(self, param_values: dict[str, Any], name: str = "") -> str | None:
        try:
            cond_met = eval(self.condition, {"__builtins__": {}}, param_values)  # noqa: S307
        except Exception:
            return None  # skip if condition can't be evaluated
        if cond_met and param_values.get(self.param) is None:
            return f"{self.param} is required when {self.condition}"
        return None


@dataclass(frozen=True)
class DependsOn:
    r"""Inter-parameter inequality, e.g. ``stride < window_size``.

    Parameters
    ----------
    expr : str
        A boolean expression over parameter names, e.g.
        ``"nr_shared_td <= latent_dim_td"``.
    description : str
        Human-readable explanation shown in error messages.
    """

    expr: str
    description: str = ""

    def validate(self, param_values: dict[str, Any], name: str = "") -> str | None:
        try:
            ok = eval(self.expr, {"__builtins__": {}}, param_values)  # noqa: S307
        except Exception:
            return None
        if not ok:
            desc = self.description or self.expr
            return f"Constraint violated: {desc}"
        return None


@dataclass(frozen=True)
class DataDependent:
    r"""Constraint that involves the input data dimensions.

    The expression is evaluated with the parameter dict **plus** extra keys
    injected from the data context (``n_samples``, ``n_channels``, …).

    Parameters
    ----------
    expr : str
        Boolean expression, e.g. ``"window_size < n_samples"``.
    description : str
        Human-readable explanation.
    """

    expr: str
    description: str = ""

    def validate(
        self,
        param_values: dict[str, Any],
        data_ctx: dict[str, Any] | None = None,
        name: str = "",
    ) -> str | None:
        if data_ctx is None:
            return None  # can't check without data context
        merged = {**param_values, **data_ctx}
        try:
            ok = eval(self.expr, {"__builtins__": {}}, merged)  # noqa: S307
        except Exception:
            return None
        if not ok:
            desc = self.description or self.expr
            return f"Data constraint violated: {desc} (data: {data_ctx})"
        return None

    def resolve_bound(
        self, param_name: str, data_ctx: dict[str, Any]
    ) -> int | float | None:
        """Try to extract an upper bound for *param_name* from the expression.

        This is a best-effort heuristic used by :func:`get_ui_hints` to
        compute effective slider bounds.  Returns ``None`` when the
        expression cannot be trivially parsed.
        """
        import re

        # Match patterns like "param < n_samples" or "param <= n_samples"
        m = re.match(
            rf"^\s*{re.escape(param_name)}\s*[<]=?\s*(\w[\w\d_]*)\s*$",
            self.expr,
        )
        if m:
            key = m.group(1)
            return data_ctx.get(key)

        # Match "param < n_samples - K"
        m = re.match(
            rf"^\s*{re.escape(param_name)}\s*[<]=?\s*(\w[\w\d_]*)\s*-\s*(\d+)\s*$",
            self.expr,
        )
        if m:
            key, offset = m.group(1), int(m.group(2))
            base = data_ctx.get(key)
            if base is not None:
                return base - offset

        # Match "param < n_samples / K" or "param < n_samples // K"
        m = re.match(
            rf"^\s*{re.escape(param_name)}\s*[<]=?\s*(\w[\w\d_]*)\s*//?\s*(\d+)\s*$",
            self.expr,
        )
        if m:
            key, divisor = m.group(1), int(m.group(2))
            base = data_ctx.get(key)
            if base is not None:
                return base // divisor

        # Match "param * K <= expr" or "param * K < expr"  (e.g. "window_size * 2 <= n_samples")
        m = re.match(
            rf"^\s*{re.escape(param_name)}\s*\*\s*(\d+)\s*[<]=?\s*(\w[\w\d_]*)\s*$",
            self.expr,
        )
        if m:
            multiplier, key = int(m.group(1)), m.group(2)
            base = data_ctx.get(key)
            if base is not None and multiplier > 0:
                return base // multiplier

        # Match "param * K <= expr - C"  (e.g. "min_size * 2 <= n_samples - 1")
        m = re.match(
            rf"^\s*{re.escape(param_name)}\s*\*\s*(\d+)\s*[<]=?\s*(\w[\w\d_]*)\s*-\s*(\d+)\s*$",
            self.expr,
        )
        if m:
            multiplier, key, offset = int(m.group(1)), m.group(2), int(m.group(3))
            base = data_ctx.get(key)
            if base is not None and multiplier > 0:
                return (base - offset) // multiplier

        return None


# ---------------------------------------------------------------------------
# ParamDef — consolidated descriptor per parameter
# ---------------------------------------------------------------------------


@dataclass
class ParamDef:
    """Full descriptor for a single hyper-parameter.

    Parameters
    ----------
    constraint : object
        One of the constraint objects above (``Interval``, ``StrOptions``, etc.)
        or a list of them.  If a list, the value must satisfy **at least one**
        (logical OR — useful for union types like ``str | int``).
    description : str
        Human-readable one-liner.
    group : str
        UI grouping hint (e.g. ``"stopping_criterion"``, ``"architecture"``).
    nullable : bool
        If ``True`` the parameter additionally accepts ``None``.
    ui_hidden : bool
        If ``True`` the demo UI should not show this parameter.
    """

    constraint: Any = None
    description: str = ""
    group: str = ""
    nullable: bool = False
    ui_hidden: bool = False

    def validate(self, value: Any, name: str = "value") -> str | None:
        """Validate *value* against the declared constraint(s)."""
        if value is None:
            if self.nullable:
                return None
            if self.constraint is None:
                return None
            # Check if any constraint explicitly allows None
            constraints = (
                self.constraint
                if isinstance(self.constraint, list)
                else [self.constraint]
            )
            for c in constraints:
                if isinstance(c, Options) and None in c.sentinels:
                    return None
            return f"{name} must not be None"

        if self.constraint is None:
            return None  # no constraint declared — anything goes

        constraints = (
            self.constraint if isinstance(self.constraint, list) else [self.constraint]
        )

        # OR semantics: value must pass at least one constraint
        errors: list[str] = []
        for c in constraints:
            err = c.validate(value, name)
            if err is None:
                return None
            errors.append(err)

        if len(errors) == 1:
            return errors[0]
        return f"{name}={value!r} does not satisfy any constraint: " + "; ".join(errors)


# ---------------------------------------------------------------------------
# Schema resolution (MRO-based, like _tags)
# ---------------------------------------------------------------------------


def get_parameter_schema(cls: type) -> dict[str, ParamDef]:
    """Resolve the full parameter schema for *cls* by walking the MRO.

    Parameters that appear in child classes override those from parents.
    The special ``_cross_constraints`` key is **merged** (lists concatenated)
    rather than overridden.

    Returns a deep copy safe to mutate.
    """
    merged: dict[str, Any] = {}
    cross: list[Any] = []

    for parent in reversed(cls.__mro__[:-2]):
        schema = getattr(parent, "_parameter_schema", None)
        if schema is None:
            continue
        for key, val in schema.items():
            if key == CROSS_CONSTRAINTS_KEY:
                cross.extend(val)
            else:
                merged[key] = val

    if cross:
        merged[CROSS_CONSTRAINTS_KEY] = cross

    return deepcopy(merged)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_params(
    instance: Any,
    data_ctx: dict[str, Any] | None = None,
) -> list[str]:
    """Validate all declared parameters of *instance*.

    Parameters
    ----------
    instance
        A detector instance (must have ``get_params`` from sklearn).
    data_ctx : dict, optional
        Data-dependent context (e.g. ``{"n_samples": 1000, "n_channels": 3}``).

    Returns
    -------
    list[str]
        A list of error messages (empty if everything is valid).
    """
    schema = get_parameter_schema(type(instance))
    if not schema:
        return []

    params = instance.get_params(deep=False)
    errors: list[str] = []

    # Per-parameter constraints
    for pname, pdef in schema.items():
        if pname == CROSS_CONSTRAINTS_KEY:
            continue
        if not isinstance(pdef, ParamDef):
            continue
        value = params.get(pname, None)
        err = pdef.validate(value, name=pname)
        if err:
            errors.append(err)

    # Cross-parameter and data-dependent constraints
    cross = schema.get(CROSS_CONSTRAINTS_KEY, [])
    for constraint in cross:
        if isinstance(constraint, (MutuallyExclusive, ConditionalRequired, DependsOn)):
            err = constraint.validate(params)
            if err:
                errors.append(err)
        elif isinstance(constraint, DataDependent):
            err = constraint.validate(params, data_ctx)
            if err:
                errors.append(err)

    return errors


# ---------------------------------------------------------------------------
# UI hint generation
# ---------------------------------------------------------------------------


def get_ui_hints(
    cls: type,
    data_ctx: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a UI-friendly representation of the parameter schema.

    For each parameter the result dict contains:

    * ``"type"`` — primary Python type (``int``, ``float``, ``str``, ``bool``).
    * ``"min"`` / ``"max"`` — effective numeric bounds (may include
      data-dependent adjustments).
    * ``"choices"`` — list of allowed string values (for ``StrOptions``).
    * ``"nullable"`` — whether ``None`` is acceptable.
    * ``"default"`` — default value from the constructor.
    * ``"description"`` — human-readable label.
    * ``"group"`` — UI grouping key.
    * ``"hidden"`` — whether to hide from the UI.

    Parameters
    ----------
    cls : type
        A detector class.
    data_ctx : dict, optional
        Optional data context for resolving data-dependent bounds.
    """
    import inspect as _inspect

    schema = get_parameter_schema(cls)
    if not schema:
        return {}

    # Grab defaults from __init__ signature
    try:
        sig = _inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        sig = None

    defaults: dict[str, Any] = {}
    if sig:
        for p in sig.parameters.values():
            if p.name == "self":
                continue
            if p.default is not _inspect.Parameter.empty:
                defaults[p.name] = p.default

    cross = schema.get(CROSS_CONSTRAINTS_KEY, [])
    hints: dict[str, dict[str, Any]] = {}

    for pname, pdef in schema.items():
        if pname == CROSS_CONSTRAINTS_KEY:
            continue
        if not isinstance(pdef, ParamDef):
            continue

        h: dict[str, Any] = {
            "description": pdef.description,
            "group": pdef.group,
            "nullable": pdef.nullable,
            "hidden": pdef.ui_hidden,
            "default": defaults.get(pname),
        }

        constraints = (
            pdef.constraint
            if isinstance(pdef.constraint, list)
            else ([pdef.constraint] if pdef.constraint else [])
        )

        # Determine primary type, bounds, choices
        for c in constraints:
            if isinstance(c, Interval):
                h["type"] = c.type.__name__
                low, high = c.effective_bounds(data_ctx)
                h.setdefault("min", low)
                h.setdefault("max", high)
                h["closed"] = c.closed.name.lower()
            elif isinstance(c, StrOptions):
                h["type"] = "str"
                h["choices"] = sorted(c.options)
            elif isinstance(c, Options):
                if isinstance(c.type, tuple):
                    h["type"] = "|".join(t.__name__ for t in c.type)
                else:
                    h["type"] = c.type.__name__
                if None in c.sentinels:
                    h["nullable"] = True
            elif isinstance(c, HasType):
                h["type"] = "|".join(t.__name__ for t in c.types)

        # Refine max bound from DataDependent cross-constraints
        if data_ctx:
            for dc in cross:
                if isinstance(dc, DataDependent):
                    resolved = dc.resolve_bound(pname, data_ctx)
                    if resolved is not None:
                        current_max = h.get("max")
                        if current_max is None or resolved < current_max:
                            h["max"] = resolved

        # Fallback type from default value
        if "type" not in h:
            default = h.get("default")
            if isinstance(default, bool):
                h["type"] = "bool"
            elif isinstance(default, int):
                h["type"] = "int"
            elif isinstance(default, float):
                h["type"] = "float"
            elif isinstance(default, str):
                h["type"] = "str"
            else:
                h["type"] = "any"

        hints[pname] = h

    return hints
