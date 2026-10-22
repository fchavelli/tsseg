"""Test parameter sets declared by the detectors.

A detector may define ``_get_test_params()`` (aeon convention): a dict, or a
list of dicts, each of which must build a valid instance with
``cls(**params)``.  A key that the constructor does not take makes that call
raise ``TypeError``, so every declared set is checked here.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from tsseg.algorithms.param_schema import validate_params


def _declared_parameter_sets(cls) -> list[dict]:
    params = cls._get_test_params()
    return params if isinstance(params, list) else [params]


def test_get_test_params_build_valid_instances(algorithm):
    """Each declared test parameter set builds an instance that validates."""
    name, cls, _ovr, _instance = algorithm
    if not hasattr(cls, "_get_test_params"):
        pytest.skip(f"{name} declares no test parameters")

    init_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    for params in _declared_parameter_sets(cls):
        unknown = set(params) - init_params
        assert not unknown, (
            f"{name}._get_test_params() names parameters that the constructor "
            f"does not take: {sorted(unknown)}"
        )
        with warnings.catch_warnings():
            # Some sets combine options on purpose (e.g. n_cps with a penalty).
            warnings.simplefilter("ignore", UserWarning)
            instance = cls(**params)
        assert validate_params(instance) == [], name
