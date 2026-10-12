"""Docstring contract: the class docstring documents the constructor.

For every detector exported by ``tsseg.algorithms``, the ``Parameters``
section of the class docstring must list exactly the parameters of
``__init__``, and every default stated as ``default=<value>`` must be the
actual default.
"""

from __future__ import annotations

import ast
import inspect
import math
import re

import pytest

import tsseg.algorithms as algorithms

DETECTORS = [name for name in algorithms.__all__ if name.endswith("Detector")]

# Detectors whose docstring is known to be incomplete, to be fixed in a later
# batch (their files are being changed elsewhere). ``strict=True``: the entry
# must go once the docstring is fixed.
KNOWN_GAPS = {
    "AutoPlaitDetector": "n_cps is not documented",
    "ClapDetector": "documents a semi_supervised parameter that does not exist",
    "ClaspDetector": "seven parameters are not documented",
    "EspressoDetector": "documents subsequence_length; the parameter is window_size",
    "FLUSSDetector": "axis is not documented",
    "HMMDetector": "two parameter lines lack the ' : ' separator",
    "InformationGainDetector": "two parameter lines lack the ' : ' separator",
    "RandomDetector": "no Parameters section",
    "TireDetector": "no Parameters section",
    "TSCP2Detector": "no Parameters section",
}

_SECTION_RE = re.compile(r"^(\w[\w ]*)\n-{3,}$", re.M)
_ARITHMETIC_RE = re.compile(r"^(?:[\d\s.+\-*/()]|sqrt)+$")
_DEFAULT_RE = re.compile(r"default\s*[=:]?\s*(.+?)\s*$")


def _parameters_section(doc: str) -> list[str] | None:
    """Lines of the ``Parameters`` section of a docstring, or ``None``."""
    doc = inspect.cleandoc(doc or "")
    headers = list(_SECTION_RE.finditer(doc))
    for i, match in enumerate(headers):
        if match.group(1) == "Parameters":
            end = headers[i + 1].start() if i + 1 < len(headers) else len(doc)
            return doc[match.end() : end].splitlines()
    return None


def _documented(doc: str) -> dict[str, str] | None:
    """Map each documented parameter to the text after ``name :``."""
    lines = _parameters_section(doc)
    if lines is None:
        return None
    entries: dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.startswith((" ", "\t")):
            continue
        names, _, spec = line.partition(" : ")
        for name in names.split(","):
            entries[name.strip().lstrip("*")] = spec.strip()
    return entries


def _constructor_parameters(cls) -> dict[str, inspect.Parameter]:
    skip = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    params = inspect.signature(cls.__init__).parameters
    return {n: p for n, p in params.items() if n != "self" and p.kind not in skip}


def _same_default(documented: str, actual) -> bool:
    text = documented.strip().strip("`")
    if text.count(")") > text.count("("):  # "(default None)"
        text = text[: text.rindex(")")].strip()
    if text.lower() in {"nan", "np.nan", "numpy.nan"}:
        return isinstance(actual, float) and math.isnan(actual)
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        if not _ARITHMETIC_RE.match(text):
            return text == str(actual)
        # An arithmetic expression such as "1/sqrt(2)" or "2 ** -0.5".
        value = eval(text, {"__builtins__": {}, "sqrt": math.sqrt})  # noqa: S307
    if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(value, actual, rel_tol=1e-12)
    return value == actual


def _load(name: str):
    try:
        return getattr(algorithms, name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(f"Cannot import {name}: {exc}")


def _params(names):
    return [
        pytest.param(
            name,
            marks=pytest.mark.xfail(reason=KNOWN_GAPS[name], strict=True),
        )
        if name in KNOWN_GAPS
        else name
        for name in names
    ]


@pytest.mark.parametrize("name", _params(DETECTORS))
def test_parameters_section_matches_signature(name):
    cls = _load(name)
    documented = _documented(cls.__doc__)
    assert documented is not None, f"{name} has no Parameters section"
    expected = set(_constructor_parameters(cls))
    assert set(documented) == expected, (
        f"{name}: undocumented {sorted(expected - set(documented))}, "
        f"documented but not in __init__ {sorted(set(documented) - expected)}"
    )


@pytest.mark.parametrize("name", DETECTORS)
def test_documented_defaults_are_actual_defaults(name):
    cls = _load(name)
    documented = _documented(cls.__doc__) or {}
    params = _constructor_parameters(cls)
    wrong = []
    for param, spec in documented.items():
        match = _DEFAULT_RE.search(spec)
        if param not in params or match is None:
            continue
        actual = params[param].default
        if actual is inspect.Parameter.empty:
            continue
        if not _same_default(match.group(1), actual):
            wrong.append(f"{param}: documented {match.group(1)}, actual {actual!r}")
    assert not wrong, f"{name}: " + "; ".join(wrong)


class TestParser:
    def test_reads_names_and_specs(self):
        doc = """Summary.

        Parameters
        ----------
        a : int, default=1
            Text.
        b, c : str
            Text.

        Returns
        -------
        int
        """
        assert _documented(doc) == {"a": "int, default=1", "b": "str", "c": "str"}

    def test_no_section(self):
        assert _documented("Summary only.") is None

    @pytest.mark.parametrize(
        "text, actual, expected",
        [
            ('"l2"', "l2", True),
            ("10", 10.0, True),
            ("None", None, True),
            ("None", 10, False),
            ("NaN", float("nan"), True),
            ("2", 1, False),
            ("(0, 10)", (0, 10), True),
            ("None)", None, True),
            ("1/sqrt(2)", 2**-0.5, True),
            ("2 ** -0.5", 2**-0.5, True),
            ("l2", "l2", True),
        ],
    )
    def test_same_default(self, text, actual, expected):
        assert _same_default(text, actual) is expected
