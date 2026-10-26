"""Behavioural tests of BeastDetector (need Rbeast >= 0.1.25)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("Rbeast")

from tsseg.algorithms import BeastDetector  # noqa: E402
from tsseg.algorithms.beast import detector as beast_module  # noqa: E402


def _step(n=1000, at=500, rng=0):
    rng = np.random.default_rng(rng)
    x = rng.normal(0, 0.3, n)
    x[at:] += 3.0
    return x


def test_change_point_is_first_index_of_new_segment():
    """Rbeast counts time from 1: the CP used to come out one sample late."""
    assert BeastDetector().fit_predict(_step()).tolist() == [500]


def test_change_point_with_deltat():
    assert BeastDetector(deltat=0.5).fit_predict(_step()).tolist() == [500]


def test_multivariate_native():
    x = _step()
    X = np.c_[x, x + np.random.default_rng(1).normal(0, 0.3, len(x))]
    assert BeastDetector().fit_predict(X).tolist() == [500]


@pytest.mark.parametrize("n_channels", [1, 2])
def test_sse_rerun_finds_the_change_point(monkeypatch, n_channels):
    """Rbeast accepts the SSE rerun of a failed run (the failure is forced here)."""
    failed = iter([True])  # the first run counts as failed, the rerun does not
    monkeypatch.setattr(beast_module, "_all_nan_output", lambda o: next(failed, False))
    x = _step()
    if n_channels == 2:
        x = np.c_[x, x + np.random.default_rng(1).normal(0, 0.3, len(x))]
    with pytest.warns(RuntimeWarning, match="cputype='sse'"):
        cps = BeastDetector().fit_predict(x)
    assert cps.tolist() == [500]


def _cpu_has_avx512():
    """Whether ``/proc/cpuinfo`` lists AVX-512 (``False`` where it is absent)."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            return "avx512f" in f.read()
    except OSError:
        return False


def test_long_univariate_series_returns():
    """Rbeast 0.1.15 never returned beyond about 4,600 points.

    Rbeast picks its numerical routines by CPU instruction set (SSE, AVX2,
    AVX-512), so a fixed seed gives the same output on one machine only
    (zhaokg/Rbeast#53). On some CI machines, Rbeast 0.1.25 returned NaN for
    every output on this series; the detector then reruns Rbeast with its SSE
    routines. If the rerun returns NaN too, the test marks the case as an
    expected failure on AVX-512 CPUs, where Rbeast is reported to compute
    wrong outputs (zhaokg/Rbeast#56), and fails anywhere else. If the rerun
    finds the change point, the test passes and says so in a warning.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cps = BeastDetector().fit_predict(_step(n=6000, at=3000))
    messages = [str(w.message) for w in caught]
    sse_rerun = any("cputype='sse'" in m for m in messages)
    nan_output = cps.size == 0 and any(
        "returned nan for ncp_median" in m for m in messages
    )
    avx512 = _cpu_has_avx512()
    if nan_output and avx512:
        pytest.xfail(
            "Rbeast returned NaN for every output on this AVX-512 CPU, also "
            f"when rerun with cputype='sse' ({sse_rerun=}; zhaokg/Rbeast#53, #56)"
        )
    assert 3000 in cps, f"{cps=}, {nan_output=}, {sse_rerun=}, {avx512=}"
    if sse_rerun:
        # Visible in the warnings summary: the record that the rerun works.
        warnings.warn(
            f"Rbeast returned NaN for every output on this CPU ({avx512=}); "
            "the rerun with cputype='sse' found the change point at 3000",
            UserWarning,
            stacklevel=1,
        )


def test_default_seed_is_reproducible():
    x = np.random.default_rng(2).normal(size=800)
    x[400:] += 1.0
    a = BeastDetector().fit_predict(x)
    b = BeastDetector().fit_predict(x)
    np.testing.assert_array_equal(a, b)
