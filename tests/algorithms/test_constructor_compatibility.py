"""Constructor migration preserves old bindings and estimator tooling."""

import inspect
import warnings

import numpy as np
import pytest

import tsseg.algorithms as algorithms
from tsseg.algorithms._dependencies import check_python_dependencies

# Frozen positional parameter order before the keyword-only migration.
LEGACY_POSITIONALS = {
    "AutoPlaitDetector": ("n_cps",),
    "BeastDetector": (
        "season",
        "tcp_minmax",
        "torder_minmax",
        "tseg_minlength",
        "scp_minmax",
        "sorder_minmax",
        "sseg_minlength",
        "period",
        "deltat",
        "mcmc_samples",
        "mcmc_burnin",
        "mcmc_chains",
        "mcmc_thin",
        "mcmc_seed",
        "cp_prob_threshold",
        "max_cps",
        "component",
        "multivariate_strategy",
        "tolerance",
        "axis",
    ),
    "BOCDDetector": (
        "hazard_lambda",
        "mu",
        "kappa",
        "alpha",
        "beta",
        "truncate",
        "cp_prob_threshold",
        "min_distance",
        "max_cps",
        "multivariate_strategy",
        "tolerance",
        "axis",
    ),
    "ChangeFinderDetector": (
        "order",
        "discount",
        "smooth_window",
        "score",
        "n_cps",
        "threshold",
        "min_distance",
        "multivariate_strategy",
        "tolerance",
        "threshold_factor",
        "min_distance_fraction",
        "consensus",
        "axis",
    ),
    "ClapDetector": (
        "window_size",
        "classifier",
        "merge_score",
        "n_splits",
        "n_jobs",
        "sample_size",
        "random_state",
        "axis",
        "n_change_points",
        "change_points",
        "n_segments",
        "distance",
    ),
    "ClaspDetector": (
        "n_segments",
        "n_change_points",
        "n_estimators",
        "window_size",
        "k_neighbours",
        "excl_radius",
        "distance",
        "score",
        "early_stopping",
        "validation",
        "threshold",
        "n_jobs",
        "axis",
    ),
    "E2USDDetector": (
        "axis",
        "window_size",
        "step",
        "n_states",
        "alpha",
        "batch_size",
        "nb_steps",
        "lr",
        "depth",
        "out_channels",
        "reduced_size",
        "kernel_size",
        "use_gpu",
        "random_state",
    ),
    "EAggloDetector": ("member", "alpha", "penalty"),
    "EspressoDetector": ("window_size", "chain_len"),
    "FLUSSDetector": (
        "window_size",
        "n_segments",
        "exclusion_factor",
        "axis",
        "multivariate_strategy",
        "tolerance",
        "threshold",
        "consensus",
        "normalize",
    ),
    "GreedyGaussianDetector": (
        "k_max",
        "lamb",
        "max_shuffles",
        "verbose",
        "random_state",
    ),
    "HdpHsmmDetector": (
        "axis",
        "alpha",
        "gamma",
        "init_state_concentration",
        "n_iter",
        "n_max_states",
        "trunc",
    ),
    "ICIDDetector": ("window_size", "alpha", "t", "psi_list", "axis"),
    "InformationGainDetector": ("k_max", "step"),
    "RandomDetector": (),
    "TiccDetector": (
        "window_size",
        "n_states",
        "lambda_parameter",
        "beta",
        "maxIters",
        "threshold",
        "num_proc",
        "max_states",
        "axis",
    ),
    "Time2StateDetector": (
        "axis",
        "window_size",
        "step",
        "n_states",
        "alpha",
        "batch_size",
        "nb_steps",
        "lr",
        "depth",
        "out_channels",
        "reduced_size",
        "kernel_size",
        "use_gpu",
        "random_state",
    ),
    "TireDetector": (
        "window_size",
        "stride",
        "domain",
        "intermediate_dim_td",
        "latent_dim_td",
        "nr_shared_td",
        "nr_ae_td",
        "loss_weight_td",
        "intermediate_dim_fd",
        "latent_dim_fd",
        "nr_shared_fd",
        "nr_ae_fd",
        "loss_weight_fd",
        "nfft",
        "norm_mode",
        "peak_distance_fraction",
        "max_epochs",
        "patience",
        "learning_rate",
    ),
}


@pytest.fixture(params=LEGACY_POSITIONALS)
def detector_class(request):
    return getattr(algorithms, request.param)


def _require_dependencies(cls):
    try:
        check_python_dependencies(cls)
    except ModuleNotFoundError as exc:
        pytest.skip(str(exc))


def test_canonical_signature_is_keyword_only_with_axis_last(detector_class):
    parameters = list(inspect.signature(detector_class).parameters.values())
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in parameters)
    assert parameters[-1].name == "axis"
    assert parameters[-1].default == 0
    assert parameters[-1].annotation in (int, "int")


def test_every_old_positional_prefix_keeps_its_binding(detector_class):
    cls = detector_class
    _require_dependencies(cls)
    parameters = inspect.signature(cls).parameters
    names = LEGACY_POSITIONALS[cls.__name__]
    values = [1 if n == "axis" else parameters[n].default for n in names]
    expected = cls(**dict(zip(names, values, strict=True)))
    for count in range(1, len(names) + 1):
        tail = dict(zip(names[count:], values[count:], strict=True))
        with pytest.warns(FutureWarning, match="keyword") as caught:
            actual = cls(*values[:count], **tail)
        assert len(caught) == 1
        assert caught[0].filename == __file__
        np.testing.assert_equal(actual.get_params(), expected.get_params())


def test_duplicate_and_excess_arguments_raise_without_a_warning(detector_class):
    cls = detector_class
    names = LEGACY_POSITIONALS[cls.__name__]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            cls(*([None] * (len(names) + 1)))
        if names:
            with pytest.raises(TypeError, match="multiple values"):
                cls(None, **{names[0]: None})
    assert not caught


def test_keyword_calls_clone_and_reset_without_positional_warnings(detector_class):
    cls = detector_class
    _require_dependencies(cls)
    kwargs = {"axis": 1}
    if "random_state" in inspect.signature(cls).parameters:
        kwargs["random_state"] = 7
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        instance = cls(**kwargs)
        parameters = instance.get_params(deep=False)
        assert parameters["axis"] == 1
        clone = instance.clone()
        np.testing.assert_equal(clone.get_params(deep=False), parameters)
        instance.reset()
        np.testing.assert_equal(instance.get_params(deep=False), parameters)
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


@pytest.mark.parametrize("name", algorithms.__all__)
def test_every_active_public_detector_has_the_canonical_signature(name):
    if name in {"HMMDetector", "PatssDetector", "KCPDDetector"}:
        pytest.skip("Deprecated detector: no signature migration")
    cls = getattr(algorithms, name)
    parameters = list(inspect.signature(cls).parameters.values())
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in parameters), name
    assert parameters[-1].name == "axis", name
