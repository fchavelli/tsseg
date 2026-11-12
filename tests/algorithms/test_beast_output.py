"""How BeastDetector reads Rbeast outputs (stand-in outputs, no Rbeast needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from tsseg.algorithms import BeastDetector
from tsseg.algorithms.beast import detector as beast_module

NAN = float("nan")


def _use_fake_rbeast(monkeypatch, output, sse_output=None, received=None):
    """Stand in for Rbeast, which need not be installed.

    ``beast`` and ``beast123`` return ``output``, or ``sse_output`` (if given)
    when called with ``cputype="sse"`` (``extra.cputype`` for ``beast123``).
    A multivariate output holds the arrays of every channel, of shape
    (slots, channels): ``"native"`` runs ``beast123`` on one channel at a
    time, in order, and each call returns the column of its channel.
    Returns the list of the ``cputype`` of each call (``None`` if not passed).
    If ``received`` is a list, the keyword arguments of each call are appended
    to it.
    """
    calls = []

    def answer(cputype):
        calls.append(cputype)
        if cputype == "sse" and sse_output is not None:
            return sse_output
        return output

    def beast(*args, **kwargs):
        if received is not None:
            received.append(kwargs)
        return answer(kwargs.get("cputype"))

    def beast123(data, extra=None, **kwargs):
        assert np.ndim(data) == 1, "beast123 gets one channel at a time"
        if received is not None:
            received.append(kwargs)
        cputype = getattr(extra, "cputype", None)
        n_runs = calls.count(cputype)  # earlier runs with the same routines
        return _channel(answer(cputype), n_runs)

    fake = SimpleNamespace(beast=beast, beast123=beast123, args=SimpleNamespace)
    monkeypatch.setattr(beast_module, "_import_rbeast", lambda: fake)
    # Skip the dependency check at construction, which looks Rbeast up.
    monkeypatch.setattr(
        BeastDetector, "_tags", {**BeastDetector._tags, "python_dependencies": None}
    )
    return calls


def _channel(output, ch):
    """The output of channel ``ch`` of a multivariate stand-in output."""
    components = {}
    for name, comp in vars(output).items():
        comp = SimpleNamespace(**vars(comp))
        if comp.cp.ndim == 2:
            comp.cp, comp.cpPr = comp.cp[:, ch], comp.cpPr[:, ch]
            comp.ncp_median = comp.ncp_median[ch : ch + 1]
        components[name] = comp
    return SimpleNamespace(**components)


def _trend(cp, cp_pr, ncp_median):
    return SimpleNamespace(
        cp=np.array(cp, dtype=float),
        cpPr=np.array(cp_pr, dtype=float),
        ncp_median=np.array(ncp_median, dtype=float),
        cpOccPr=np.zeros(1000),
    )


def test_ncp_median_caps_the_ranked_change_points(monkeypatch):
    output = SimpleNamespace(trend=_trend([300, 700, NAN], [0.9, 0.8, NAN], [1.0]))
    calls = _use_fake_rbeast(monkeypatch, output)
    assert BeastDetector().fit_predict(np.zeros(1000)).tolist() == [300]
    assert calls == [None]  # one run, with Rbeast's default routines


def test_nan_ncp_median_does_not_crash_univariate(monkeypatch):
    """Rbeast can return NaN for ``ncp_median`` (seen in CI on 6,000 points)."""
    output = SimpleNamespace(trend=_trend([300, 700, NAN], [0.9, 0.8, NAN], [NAN]))
    calls = _use_fake_rbeast(monkeypatch, output)
    with pytest.warns(RuntimeWarning, match="ncp_median"):
        cps = BeastDetector().fit_predict(np.zeros(1000))
    assert cps.tolist() == [300, 700]
    # NaN in the only ncp_median counts as a failed run: one SSE rerun, which
    # returns the same output here.
    assert calls == [None, "sse"]


def test_nan_ncp_median_does_not_crash_multivariate(monkeypatch):
    output = SimpleNamespace(
        trend=_trend(
            [[300, 300], [700, 700], [NAN, NAN]],
            [[0.9, 0.9], [0.8, 0.8], [NAN, NAN]],
            [1.0, NAN],
        )
    )
    calls = _use_fake_rbeast(monkeypatch, output)
    with pytest.warns(RuntimeWarning, match="ncp_median"):
        cps = BeastDetector().fit_predict(np.zeros((1000, 2)))
    # Channel 0 is capped at one CP by its ncp_median, channel 1 is not.
    assert cps.tolist() == [300, 700]
    # A finite ncp_median in one channel is not a failed run: no rerun.
    assert calls == [None, None]  # one run per channel


def _all_nan_output(n_channels=1):
    """A failed Rbeast run: NaN everywhere (seen in CI on some CPUs)."""
    if n_channels == 1:
        trend = _trend([NAN, NAN], [NAN, NAN], [NAN])
    else:
        nan = [[NAN] * n_channels] * 2
        trend = _trend(nan, nan, [NAN] * n_channels)
    trend.cpOccPr = np.full(1000, NAN)
    return SimpleNamespace(trend=trend)


def test_all_nan_output_is_rerun_with_sse_univariate(monkeypatch):
    """A run whose outputs are all NaN is rerun once with Rbeast's SSE routines."""
    sse_output = SimpleNamespace(trend=_trend([300, NAN], [0.9, NAN], [1.0]))
    calls = _use_fake_rbeast(monkeypatch, _all_nan_output(), sse_output)
    with pytest.warns(RuntimeWarning, match="cputype='sse'"):
        cps = BeastDetector().fit_predict(np.zeros(1000))
    assert cps.tolist() == [300]
    assert calls == [None, "sse"]


def test_all_nan_output_is_rerun_with_sse_multivariate(monkeypatch):
    sse_output = SimpleNamespace(
        trend=_trend([[300, 700], [NAN, NAN]], [[0.9, 0.9], [NAN, NAN]], [1.0, 1.0])
    )
    calls = _use_fake_rbeast(monkeypatch, _all_nan_output(n_channels=2), sse_output)
    with pytest.warns(RuntimeWarning, match="cputype='sse'"):
        # max_cps=2: one candidate per channel, the count without it
        cps = BeastDetector(max_cps=2).fit_predict(np.zeros((1000, 2)))
    assert cps.tolist() == [300, 700]
    assert calls == [None, None, "sse", "sse"]  # one run per channel


def test_all_nan_output_gives_no_change_point(monkeypatch):
    """NaN everywhere, also in the SSE rerun: no CP, and both warnings."""
    calls = _use_fake_rbeast(monkeypatch, _all_nan_output())
    with pytest.warns(RuntimeWarning) as record:
        cps = BeastDetector().fit_predict(np.zeros(1000))
    assert cps.size == 0
    assert cps.dtype == np.int64
    assert calls == [None, "sse"]  # a single rerun
    messages = [str(w.message) for w in record]
    assert any("cputype='sse'" in m for m in messages), messages
    assert any("ncp_median" in m for m in messages), messages


_PRIOR_PARAMS = dict(
    period=24.0,
    tcp_minmax=(1, 3),
    torder_minmax=(0, 2),
    tseg_minlength=40,
    scp_minmax=(0, 4),
    sorder_minmax=(1, 3),
    sseg_minlength=30,
)


@pytest.mark.parametrize("season", ["none", "harmonic"])
def test_native_passes_the_prior_to_beast123(monkeypatch, season):
    """``beast123`` takes the prior as an object: ``"native"`` passed none.

    The fields are those ``Rbeast.beast`` fills from the same arguments, the
    seasonal ones only when there is a seasonal component.
    """
    output = SimpleNamespace(
        trend=_trend([[300, 300], [NAN, NAN]], [[0.9, 0.9], [NAN, NAN]], [1.0, 1.0])
    )
    received = []
    _use_fake_rbeast(monkeypatch, output, received=received)
    detector = BeastDetector(season=season, **_PRIOR_PARAMS)
    assert detector.fit_predict(np.zeros((1000, 2))).tolist() == [300]
    expected = {
        "trendMinKnotNum": 1,
        "trendMaxKnotNum": 3,
        "trendMinOrder": 0,
        "trendMaxOrder": 2,
        "trendMinSepDist": 40,
    }
    if season != "none":
        expected |= {
            "seasonMinKnotNum": 0,
            "seasonMaxKnotNum": 4,
            "seasonMinOrder": 1,
            "seasonMaxOrder": 3,
            "seasonMinSepDist": 30,
        }
    assert len(received) == 2  # one run per channel
    assert all(vars(kwargs["prior"]) == expected for kwargs in received)


@pytest.mark.parametrize(
    ("params", "trend", "season"),
    [
        ({}, (0, 10), (0, 10)),
        ({"max_cps": 5}, (0, 10), (0, 10)),
        ({"max_cps": 15}, (0, 15), (0, 10)),
        ({"max_cps": 15, "season": "harmonic", "period": 24.0}, (0, 15), (0, 10)),
        (
            {"max_cps": 15, "season": "harmonic", "period": 24.0, "component": "both"},
            (0, 15),
            (0, 15),
        ),
        (
            {
                "max_cps": 15,
                "season": "harmonic",
                "period": 24.0,
                "component": "season",
            },
            (0, 10),
            (0, 15),
        ),
    ],
)
def test_max_cps_raises_the_maximum_number_of_change_points(
    monkeypatch, params, trend, season
):
    """Rbeast returns at most the maximum of ``tcp_minmax`` (or ``scp_minmax``)
    change points per component and channel: a larger ``max_cps`` could not be
    reached, on 29 % of the Guided runs of a 47-dataset smoke test."""
    output = SimpleNamespace(
        trend=_trend([[300, 300], [NAN, NAN]], [[0.9, 0.9], [NAN, NAN]], [1.0, 1.0])
    )
    received = []
    _use_fake_rbeast(monkeypatch, output, received=received)
    BeastDetector(**params).fit_predict(np.zeros((1000, 2)))  # "native"
    BeastDetector(multivariate_strategy="l2", **params).fit_predict(np.zeros((1000, 2)))
    native, univariate = received[0], received[-1]  # 2 channels, then 1 run
    assert (native["prior"].trendMinKnotNum, native["prior"].trendMaxKnotNum) == trend
    assert tuple(univariate["tcp_minmax"]) == trend
    assert tuple(univariate["scp_minmax"]) == season
    if params.get("season", "none") != "none":
        prior = native["prior"]
        assert (prior.seasonMinKnotNum, prior.seasonMaxKnotNum) == season
