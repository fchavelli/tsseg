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
@pytest.mark.parametrize("strategy", ["ensembling", "l2"])
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
    assert calls == [normalize] * (2 if strategy == "ensembling" else 1)


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
