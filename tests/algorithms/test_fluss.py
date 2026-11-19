"""FLUSSDetector: CAC threshold (no known number of segments) and output order."""

import numpy as np
import pytest

pytest.importorskip("stumpy")

from tsseg.algorithms import FLUSSDetector  # noqa: E402


def _regimes(rng=0):
    rng = np.random.default_rng(rng)
    t = np.arange(1000)
    x = np.concatenate([np.sin(t / 5), np.sign(np.sin(t / 7)), np.sin(t / 3)])
    return x + 0.1 * rng.normal(size=x.size)


def test_threshold_zero_finds_nothing():
    det = FLUSSDetector(window_size=20, n_segments=None, threshold=0.0)
    assert len(det.fit_predict(_regimes())) == 0


def test_threshold_finds_the_regime_changes():
    det = FLUSSDetector(window_size=20, n_segments=None, threshold=0.2)
    pred = det.fit_predict(_regimes())
    assert len(pred) >= 2
    for cp in (1000, 2000):
        assert np.min(np.abs(pred - cp)) <= 30
    assert np.all(det.profile[pred] < 0.2)
    assert np.all(np.diff(pred) >= 5 * 20)


def test_known_number_of_segments_is_unchanged():
    pred = FLUSSDetector(window_size=20, n_segments=3).fit_predict(_regimes())
    assert len(pred) == 2


def test_ensembling_without_k():
    x = _regimes()
    rng = np.random.default_rng(1)
    X = np.column_stack(
        [x, x + 0.05 * rng.normal(size=x.size), rng.normal(size=x.size)]
    )
    det = FLUSSDetector(window_size=20, n_segments=None, threshold=0.2, consensus=0.5)
    pred = det.fit_predict(X)
    for cp in (1000, 2000):
        assert np.min(np.abs(pred - cp)) <= 40


@pytest.mark.parametrize("n_segments", [4, 5])
def test_known_number_of_segments_returns_sorted_change_points(n_segments):
    # stumpy.fluss returns the regime locations in extraction order (lowest
    # corrected arc curve first), e.g. [981, 1982, 880] for 4 segments here.
    det = FLUSSDetector(window_size=20, n_segments=n_segments)
    pred = det.fit_predict(_regimes())
    assert len(pred) == n_segments - 1
    assert np.all(np.diff(pred) > 0)
    np.testing.assert_array_equal(det.scores, det.profile[pred])


def test_l2_strategy_returns_sorted_change_points():
    x = _regimes()
    X = np.column_stack([x, 0.5 * x])
    det = FLUSSDetector(window_size=20, n_segments=4, multivariate_strategy="l2")
    pred = det.fit_predict(X)
    assert len(pred) == 3
    assert np.all(np.diff(pred) > 0)


def _assert_change_points(pred, n):
    assert isinstance(pred, np.ndarray)
    assert np.issubdtype(pred.dtype, np.integer)
    if pred.size:
        assert pred.min() >= 1 and pred.max() <= n - 1
    assert np.all(np.diff(pred) > 0)


def test_flat_corrected_arc_curve_gives_no_change_point_at_0():
    # On white noise, the corrected arc curve (CAC) is 1 everywhere but around
    # 681. Once that point is masked, stumpy's regime extraction takes the
    # argmin of a flat curve, 0, and returned [0, 681].
    x = np.random.default_rng(0).normal(size=1000)
    det = FLUSSDetector(window_size=50, n_segments=3)
    pred = det.fit_predict(x)
    _assert_change_points(pred, x.size)
    np.testing.assert_array_equal(pred, [681])
    assert np.all(det.profile[pred] < 1.0)


def test_more_segments_than_the_curve_holds_returns_fewer_change_points():
    # 29 extractions mask the whole CAC (100 points on each side of each
    # change point): the last ones returned 0 again and again.
    det = FLUSSDetector(window_size=20, n_segments=30)
    x = _regimes()
    pred = det.fit_predict(x)
    _assert_change_points(pred, x.size)
    assert 2 <= len(pred) < 29
    assert np.all(det.profile[pred] < 1.0)
    np.testing.assert_array_equal(det.scores, det.profile[pred])


def test_ensembling_drops_the_change_points_at_0_of_the_channels():
    # Each channel returned 0 several times; their vote kept a change point
    # at 0.
    x = _regimes()
    rng = np.random.default_rng(1)
    X = np.column_stack([x, x + 0.05 * rng.normal(size=x.size)])
    pred = FLUSSDetector(window_size=20, n_segments=30).fit_predict(X)
    _assert_change_points(pred, x.size)


def test_one_segment_returns_an_empty_array():
    pred = FLUSSDetector(n_segments=1).fit_predict(_regimes())
    _assert_change_points(pred, 3000)
    assert pred.size == 0


def _mean_shifts(rng=0):
    # Three segments that differ by their mean only: white noise around -0.8,
    # 0.5 and -0.1, change points at 300 and 700.
    rng = np.random.default_rng(rng)
    means = np.repeat([-0.8, 0.5, -0.1], [300, 400, 300])
    return means + 0.2 * rng.normal(size=means.size)


def _near(pred, cps, margin):
    return [bool(pred.size) and np.min(np.abs(pred - cp)) <= margin for cp in cps]


def test_default_matrix_profile_is_z_normalized():
    # Z-normalized windows of white noise look alike whatever their mean: the
    # default (the matrix profile of the paper) misses these change points.
    det = FLUSSDetector(window_size=10, n_segments=3)
    assert det.get_params()["normalize"] is True
    pred = det.fit_predict(_mean_shifts())
    assert not all(_near(pred, (300, 700), 20))


@pytest.mark.parametrize("window_size", [10, 20, 50])
def test_non_normalized_matrix_profile_finds_shifts_in_mean(window_size):
    det = FLUSSDetector(window_size=window_size, n_segments=3, normalize=False)
    pred = det.fit_predict(_mean_shifts())
    _assert_change_points(pred, 1000)
    assert len(pred) == 2
    assert all(_near(pred, (300, 700), 40))


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("strategy", ["ensembling", "l2", "cac_mean"])
def test_normalize_is_passed_to_the_matrix_profile(monkeypatch, normalize, strategy):
    import stumpy

    calls = []
    stump = stumpy.stump

    def spy(*args, **kwargs):
        calls.append(kwargs.get("normalize"))
        return stump(*args, **kwargs)

    monkeypatch.setattr(stumpy, "stump", spy)
    x = _mean_shifts()
    X = np.column_stack([x, -x])
    FLUSSDetector(
        window_size=20,
        n_segments=3,
        normalize=normalize,
        multivariate_strategy=strategy,
    ).fit_predict(X)
    assert calls == [normalize] * (1 if strategy == "l2" else 2)


def test_normalize_without_a_known_number_of_segments():
    # On raw windows the arcs stay in their segment, so the corrected arc
    # curve is low over whole segments: the default threshold of 0.5 keeps
    # minima inside them, a lower one keeps the two change points only.
    x = _mean_shifts()
    loose = FLUSSDetector(window_size=20, n_segments=None, normalize=False)
    assert len(loose.fit_predict(x)) > 2
    det = FLUSSDetector(window_size=20, n_segments=None, normalize=False, threshold=0.1)
    pred = det.fit_predict(x)
    _assert_change_points(pred, 1000)
    assert len(pred) == 2
    assert all(_near(pred, (300, 700), 40))


def test_normalize_must_be_a_bool():
    with pytest.raises(ValueError, match="normalize"):
        FLUSSDetector(normalize="no").fit(_mean_shifts())


# multivariate_strategy="cac_mean": mean of the corrected arc curves of the
# channels (Gharghabi et al., DMKD 2019, section 3.4), then one regime
# extraction on that curve.


def _channels(rng=0):
    # Two channels with the regime changes of _regimes (1000, 2000) under
    # different noise, and one channel of white noise.
    rng = np.random.default_rng(rng)
    x = _regimes()
    return np.column_stack(
        [x, 2 * x + 0.3 * rng.normal(size=x.size), rng.normal(size=x.size)]
    )


def _channel_cac(x, window_size=20, exclusion_factor=5, normalize=True):
    import stumpy

    mp = stumpy.stump(x, m=window_size, normalize=normalize)
    cac, _ = stumpy.fluss(
        mp[:, 1], L=window_size, excl_factor=exclusion_factor, n_regimes=1
    )
    return cac


def _mean_cac(X, **kwargs):
    return np.mean([_channel_cac(X[:, c], **kwargs) for c in range(X.shape[1])], 0)


def _cac_mean(**kwargs):
    return FLUSSDetector(window_size=20, multivariate_strategy="cac_mean", **kwargs)


def test_cac_mean_profile_is_the_mean_of_the_channel_curves():
    X = _channels()
    det = _cac_mean(n_segments=3)
    det.fit_predict(X)
    expected = _mean_cac(X)
    assert det.profile.shape == (X.shape[0] - 20 + 1,)
    np.testing.assert_allclose(det.profile, expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(det.profiles, det.profile)


@pytest.mark.parametrize("n_segments", [3, 5, 30])
def test_cac_mean_known_number_of_segments_follows_the_regime_extraction(n_segments):
    # Same regime extraction (REA) as stumpy.fluss, on the mean curve; the
    # points it takes on a flat curve (1 everywhere) are dropped.
    from stumpy.floss import _rea

    X = _channels()
    det = _cac_mean(n_segments=n_segments)
    pred = det.fit_predict(X)
    cac = det.profile
    locs = _rea(cac, n_segments, 20, excl_factor=5)
    np.testing.assert_array_equal(pred, np.sort(locs[cac[locs] < 1.0]))
    _assert_change_points(pred, X.shape[0])
    assert len(pred) <= n_segments - 1
    np.testing.assert_array_equal(det.scores, cac[pred])
    np.testing.assert_array_equal(det.predict_scores(X), cac[pred])


def test_cac_mean_finds_the_change_points_shared_by_the_channels():
    pred = _cac_mean(n_segments=3).fit_predict(_channels())
    assert len(pred) == 2
    assert all(_near(pred, (1000, 2000), 30))


def test_cac_mean_threshold_without_a_known_number_of_segments():
    # The noise channel, whose curve stays near 1, lifts the mean: about 0.37
    # at the change points (0.01 and 0.12 in the other channels), 0.45 at
    # least elsewhere.
    X = _channels()
    det = _cac_mean(n_segments=None, threshold=0.42)
    pred = det.fit_predict(X)
    cac, zone = _mean_cac(X), 5 * 20
    expected, masked = [], cac.copy()
    while masked.min() < 0.42:
        loc = int(np.argmin(masked))
        expected.append(loc)
        masked[max(loc - zone, 0) : loc + zone] = 1.0
    np.testing.assert_array_equal(pred, sorted(expected))
    assert len(pred) == 2
    assert all(_near(pred, (1000, 2000), 30))
    assert np.all(det.profile[pred] < 0.42)
    assert np.all(np.diff(pred) >= zone)


@pytest.mark.parametrize("n_segments", [4, None])
def test_cac_mean_flat_curve_gives_no_change_point(monkeypatch, n_segments):
    # A corrected arc curve at 1 everywhere: no point is crossed by fewer arcs
    # than expected, so no change point, even with a number of segments.
    import stumpy

    def flat(index, L, n_regimes, excl_factor):
        return np.ones(index.shape[0]), np.zeros(n_regimes - 1, dtype=np.int64)

    monkeypatch.setattr(stumpy, "fluss", flat)
    det = _cac_mean(n_segments=n_segments, threshold=1.0)
    pred = det.fit_predict(_channels())
    _assert_change_points(pred, 3000)
    assert pred.size == 0
    assert det.scores.size == 0


@pytest.mark.parametrize("n_segments", [4, None])
def test_cac_mean_on_copies_of_a_channel_is_univariate_fluss(n_segments):
    x = _regimes()
    uni = FLUSSDetector(window_size=20, n_segments=n_segments, threshold=0.3)
    expected = uni.fit_predict(x)
    det = _cac_mean(n_segments=n_segments, threshold=0.3)
    pred = det.fit_predict(np.column_stack([x, x]))
    np.testing.assert_array_equal(pred, expected)
    np.testing.assert_array_equal(det.profile, uni.profile)


@pytest.mark.parametrize("strategy", ["ensembling", "l2"])
def test_cac_mean_univariate_is_unchanged(strategy):
    x = _regimes()
    for n_segments in (4, None):
        ref = FLUSSDetector(
            window_size=20, n_segments=n_segments, multivariate_strategy=strategy
        )
        det = _cac_mean(n_segments=n_segments)
        np.testing.assert_array_equal(det.fit_predict(x), ref.fit_predict(x))
        np.testing.assert_array_equal(det.profile, ref.profile)


def test_cac_mean_ignores_tolerance_and_consensus():
    X = _channels()
    ref = _cac_mean(n_segments=None, threshold=0.42).fit_predict(X)
    assert len(ref) == 2
    for tolerance, consensus in [(0.0, 0.0), (0.2, 1.0), (50.0, 0.5)]:
        det = _cac_mean(
            n_segments=None, threshold=0.42, tolerance=tolerance, consensus=consensus
        )
        np.testing.assert_array_equal(det.fit_predict(X), ref)


def test_cac_mean_reads_channels_on_either_axis():
    X = _channels()
    ref = _cac_mean(n_segments=3).fit_predict(X)
    det = FLUSSDetector(
        window_size=20, n_segments=3, multivariate_strategy="cac_mean", axis=1
    )
    np.testing.assert_array_equal(det.fit_predict(X.T), ref)


def test_cac_mean_without_normalization():
    # The raw subsequences see shifts in level in every channel.
    x = _mean_shifts()
    X = np.column_stack([x, 3 * x + 1])
    det = FLUSSDetector(
        window_size=20, n_segments=3, normalize=False, multivariate_strategy="cac_mean"
    )
    pred = det.fit_predict(X)
    np.testing.assert_allclose(
        det.profile, _mean_cac(X, normalize=False), rtol=0, atol=1e-12
    )
    assert all(_near(pred, (300, 700), 40))


def test_unknown_multivariate_strategy_is_refused():
    with pytest.raises(ValueError, match="multivariate_strategy"):
        FLUSSDetector(multivariate_strategy="mean").fit(_channels())
