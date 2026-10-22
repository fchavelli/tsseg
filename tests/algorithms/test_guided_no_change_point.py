"""Guided mode on a series without any change point.

A benchmark harness running detectors in guided (semi-supervised) mode passes
the true number of change points or states to the constructor.  On a series
made of a single regime this is ``n_cps=0`` (``n_segments=1``) or
``n_states=1``, and a detector that advertises
``capability:semi_supervised`` must accept it instead of failing its
parameter validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms.param_schema import validate_params

from .conftest import SUPERVISION_PARAMS, apply_supervision

# Guided detectors that take no count: the guidance is something else.
_GUIDED_WITHOUT_COUNT = {
    "HMMDetector": "the number of states is the number of emission functions",
}


# Detectors whose schema still refuses a single-regime guidance, because the
# detector itself has not been checked with it yet.
# Empty: every guided detector is checked by a test of this module.
_NOT_YET_CHECKED: dict[str, str] = {}


def _supervision_errors(instance, n_samples: int) -> list[str]:
    errors = validate_params(
        instance, data_ctx={"n_samples": n_samples, "n_channels": 1}
    )
    return [
        err
        for err in errors
        if any(err.startswith(f"{name}=") for name in SUPERVISION_PARAMS)
    ]


def test_supervision_sets_a_constructor_parameter(algorithm):
    """``apply_supervision`` guides every detector that has a guided mode.

    Otherwise the guided tests of this module and of the generic suite run
    the detector with its unsupervised defaults.
    """
    name, _cls, _ovr, instance = algorithm
    if not instance.get_tag(
        "capability:semi_supervised", raise_error=False, tag_value_default=False
    ):
        pytest.skip(f"{name} has no guided mode")
    if name in _GUIDED_WITHOUT_COUNT:
        pytest.skip(_GUIDED_WITHOUT_COUNT[name])
    # Two change points, three segments, two states: each count differs.
    labels = np.repeat([0, 1, 0], 10)
    expected = {"change_points": 2, "segments": 3, "states": 2}
    apply_supervision(instance, labels)
    guided = {
        param: value
        for param, value in instance.get_params(deep=False).items()
        if param in SUPERVISION_PARAMS
    }
    assert guided, f"{name}: no constructor parameter is set by apply_supervision"
    assert guided == {param: expected[SUPERVISION_PARAMS[param]] for param in guided}


def test_schema_accepts_single_regime_guidance(algorithm, request):
    """Supervision derived from a constant label sequence passes validation."""
    name, _cls, _ovr, instance = algorithm
    if not instance.get_tag(
        "capability:semi_supervised", raise_error=False, tag_value_default=False
    ):
        pytest.skip(f"{name} has no guided mode")
    if name in _NOT_YET_CHECKED:
        request.applymarker(
            pytest.mark.xfail(reason=_NOT_YET_CHECKED[name], strict=True)
        )
    n_samples = 1_000
    apply_supervision(instance, np.zeros(n_samples, dtype=int))
    assert _supervision_errors(instance, n_samples) == []


def _two_regime_series(n_samples: int = 600, seed: int = 0) -> np.ndarray:
    """A clear mean shift halfway: guidance, not the data, must rule out a CP."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=n_samples)
    X[n_samples // 2 :] += 5.0
    return X


def test_clasp_guided_zero_change_points():
    pytest.importorskip("numba")
    from tsseg.algorithms import ClaspDetector

    X = _two_regime_series()
    for kwargs in ({"n_change_points": 0}, {"n_segments": 1}):
        detector = ClaspDetector(window_size=10, **kwargs)
        cps = detector.fit_predict(X)
        assert isinstance(cps, np.ndarray)
        assert cps.size == 0, kwargs


def test_clap_guided_zero_change_points():
    pytest.importorskip("numba")
    pytest.importorskip("aeon")
    from tsseg.algorithms import ClapDetector

    X = _two_regime_series()
    for kwargs in ({"n_change_points": 0}, {"n_segments": 1}):
        detector = ClapDetector(window_size=10, **kwargs)
        labels = detector.fit_predict(X)
        assert labels.shape == (X.shape[0],)
        assert np.unique(labels).size == 1, kwargs


def test_time2state_guided_single_state():
    pytest.importorskip("torch")
    from tsseg.algorithms import Time2StateDetector

    X = _two_regime_series()
    detector = Time2StateDetector(
        n_states=1, window_size=32, step=16, nb_steps=2, random_state=0
    )
    labels = detector.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert np.unique(labels).size == 1


def test_e2usd_guided_single_state():
    pytest.importorskip("torch")
    from tsseg.algorithms import E2USDDetector

    X = _two_regime_series()
    detector = E2USDDetector(
        n_states=1, window_size=32, step=16, nb_steps=2, random_state=0
    )
    labels = detector.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert np.unique(labels).size == 1


def test_espresso_guided_single_segment():
    from tsseg.algorithms import EspressoDetector

    X = _two_regime_series()
    detector = EspressoDetector(window_size=10, n_segments=1, random_state=0)
    cps = detector.fit_predict(X)
    assert isinstance(cps, np.ndarray)
    assert cps.size == 0


def test_ticc_guided_single_state():
    from tsseg.algorithms import TiccDetector

    X = _two_regime_series(n_samples=300)
    detector = TiccDetector(window_size=3, n_states=1, maxIters=5)
    labels = detector.fit_predict(X)
    assert labels.shape == (X.shape[0],)
    assert np.unique(labels).size == 1


def test_tscp2_guided_zero_change_points():
    pytest.importorskip("tensorflow")
    pytest.importorskip("tcn")
    from tsseg.algorithms import TSCP2Detector

    X = _two_regime_series(n_samples=300)
    detector = TSCP2Detector(
        n_cps=0, window_size=16, epochs=2, batch_size=16, random_state=0
    )
    cps = detector.fit_predict(X)
    assert isinstance(cps, np.ndarray)
    assert cps.size == 0


@pytest.mark.parametrize(
    ("n_channels", "strategy"), [(1, "l2"), (2, "l2"), (2, "ensembling")]
)
def test_bocd_guided_zero_change_points(n_channels, strategy):
    from tsseg.algorithms import BOCDDetector

    x = _two_regime_series(n_samples=300)
    X = x if n_channels == 1 else np.c_[x, _two_regime_series(300, seed=1)]
    params = {"multivariate_strategy": strategy, "cp_prob_threshold": 0.02}
    # The data alone yield a change point: the guidance must rule it out.
    assert BOCDDetector(max_cps=1, **params).fit_predict(X).size == 1
    cps = BOCDDetector(max_cps=0, **params).fit_predict(X)
    assert isinstance(cps, np.ndarray)
    assert cps.size == 0


@pytest.mark.parametrize(
    ("n_channels", "strategy"),
    [(1, "native"), (2, "native"), (2, "l2"), (2, "ensembling")],
)
def test_beast_guided_zero_change_points(n_channels, strategy):
    pytest.importorskip("Rbeast")
    from tsseg.algorithms import BeastDetector

    x = _two_regime_series(n_samples=300)
    X = x if n_channels == 1 else np.c_[x, _two_regime_series(300, seed=1)]
    params = {"multivariate_strategy": strategy, "mcmc_seed": 1}
    assert BeastDetector(max_cps=1, **params).fit_predict(X).size == 1
    cps = BeastDetector(max_cps=0, **params).fit_predict(X)
    assert isinstance(cps, np.ndarray)
    assert cps.size == 0


def test_ggs_guided_zero_change_points():
    from tsseg.algorithms import GreedyGaussianDetector

    X = _two_regime_series(n_samples=300)
    assert GreedyGaussianDetector(k_max=1, random_state=0).fit_predict(X).size == 1
    cps = GreedyGaussianDetector(k_max=0, random_state=0).fit_predict(X)
    assert isinstance(cps, np.ndarray)
    assert cps.size == 0
