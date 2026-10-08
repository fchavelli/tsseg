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

from .conftest import apply_supervision

# Constructor parameters that ``apply_supervision`` may set from the labels.
_SUPERVISION_PARAMS = (
    "n_cps",
    "n_change_points",
    "n_bkps",
    "n_breakpoints",
    "N",
    "n_segments",
    "n_states",
    "n_regimes",
    "n_clusters",
    "n_components",
    "K",
)


# Detectors whose schema still refuses a single-regime guidance: the detector
# itself has not been run with it yet (it needs TensorFlow or a C binary).
_NOT_YET_CHECKED = {
    "AutoPlaitDetector": "n_cps=0 not yet checked against the AutoPlait binary",
    "TSCP2Detector": "n_cps=0 not yet checked against the TensorFlow model",
}


def _supervision_errors(instance, n_samples: int) -> list[str]:
    errors = validate_params(
        instance, data_ctx={"n_samples": n_samples, "n_channels": 1}
    )
    return [
        err
        for err in errors
        if any(err.startswith(f"{name}=") for name in _SUPERVISION_PARAMS)
    ]


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
