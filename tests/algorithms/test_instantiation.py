import pytest


def test_algorithm_instantiation_default(algorithm_entry):
    info = algorithm_entry
    cls = info["cls"]
    config = info["config"]

    if config.init_with_defaults:
        instance = cls()
    else:
        kwargs = config.init_kwargs or {}
        instance = cls(**kwargs)

    assert isinstance(instance, cls), f"Failed to instantiate {info['name']}"


@pytest.mark.parametrize("use_fit_params", [False, True])
def test_algorithm_instantiation_with_overrides(algorithm_entry, use_fit_params):
    info = algorithm_entry
    cls = info["cls"]
    config = info["config"]

    if use_fit_params:
        if config.fit_init_kwargs is None:
            pytest.skip("No specific fit-time kwargs defined")
        instance = cls(**config.fit_init_kwargs)
    else:
        kwargs = config.init_kwargs or {}
        if not kwargs:
            pytest.skip("No override kwargs defined")
        instance = cls(**kwargs)

    assert isinstance(instance, cls)
