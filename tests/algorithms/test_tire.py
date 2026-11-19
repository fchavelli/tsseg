"""Change point selection of TireDetector (stride, peak distance, threshold)."""

import numpy as np
import pytest

pytest.importorskip("torch")

from scipy.signal import peak_prominences  # noqa: E402

from tsseg.algorithms import TireDetector  # noqa: E402

SMALL = {
    "window_size": 16,
    "max_epochs": 3,
    "patience": 2,
    "intermediate_dim_td": 8,
    "latent_dim_td": 2,
    "nr_shared_td": 1,
    "nr_ae_td": 2,
    "domain": "TD",
    "random_state": 0,
}


def _steps(n_per=150, rng=0):
    rng = np.random.default_rng(rng)
    return np.concatenate([rng.normal(m, 0.3, n_per) for m in (0.0, 2.0, -1.0, 1.5)])


def test_peak_distance_is_a_fraction_of_the_length():
    x = _steps()
    det = TireDetector(**SMALL, peak_distance_fraction=0.1, prominence_threshold=0.0)
    pred = det.fit_predict(x)
    assert len(pred) > 0
    assert np.all(np.diff(pred) >= 0.1 * len(x))


def test_prominence_threshold():
    x = _steps()
    det = TireDetector(**SMALL, prominence_threshold=0.5)
    pred = det.fit_predict(x)
    prominences = peak_prominences(det.change_scores_, pred)[0]
    assert np.all(prominences >= 0.5)
    loose = TireDetector(**SMALL, prominence_threshold=0.0).fit_predict(x)
    assert len(loose) >= len(pred)


def test_guided_returns_n_segments_minus_one():
    pred = TireDetector(**SMALL, n_segments=4).fit_predict(_steps())
    assert len(pred) == 3


@pytest.mark.parametrize("stride", [2, 4])
def test_stride_returns_time_points(stride):
    """With ``stride > 1`` the change points are time points, not windows.

    The compared windows are ``window_size * stride`` points apart, so a step
    change gives a plateau of ``window_size * (stride - 1)`` points in the
    dissimilarity: the tolerance grows with the stride.
    """
    x = _steps()
    det = TireDetector(**SMALL, stride=stride, n_segments=4)
    pred = det.fit_predict(x)
    assert det.change_scores_.shape == (len(x),)
    assert len(pred) == 3
    tolerance = SMALL["window_size"] * stride / 2
    assert np.all(np.abs(pred - np.array([150, 300, 450])) <= tolerance)


def test_stride_change_points_are_peaks_of_the_scores():
    x = _steps()
    det = TireDetector(**SMALL, stride=3, prominence_threshold=0.2)
    pred = det.fit_predict(x)
    assert len(pred) > 0
    prominences = peak_prominences(det.change_scores_, pred)[0]
    assert np.all(prominences >= 0.2)


def test_single_segment_returns_no_change_point():
    """Guided with one segment: no change point, whatever the scores."""
    det = TireDetector(**SMALL, n_segments=1, prominence_threshold=0.0)
    pred = det.fit_predict(_steps())
    assert isinstance(pred, np.ndarray)
    assert pred.size == 0
    assert det.change_scores_.shape == (600,)


# ---------------------------------------------------------------------------
# Fidelity to the paper (De Ryck et al. 2021): per-channel rescaling (§III-B),
# per-channel DFT (eq. 3), smoothing of the representations in every domain
# (eq. 11). The autoencoders are replaced by a stub that records their input,
# so these tests check the pipeline around them without training.
# ---------------------------------------------------------------------------

from tsseg.algorithms.tire import detector as tire_detector  # noqa: E402
from tsseg.algorithms.tire import utils as tire_utils  # noqa: E402


def _record_training(monkeypatch):
    """Replace the training by fixed random features; return the inputs seen."""
    seen = []

    def fake_train(self, windows, *args):
        seen.append(np.array(windows))
        rng = np.random.default_rng(len(seen))
        return rng.normal(size=(windows.shape[0], 1))

    monkeypatch.setattr(TireDetector, "_train_autoencoder", fake_train)
    return seen


def _rescaled(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


def _multichannel(n=300, rng=0):
    rng = np.random.default_rng(rng)
    t = np.arange(n)
    return np.column_stack(
        [
            50 + 100 * rng.normal(size=n),
            np.where(t < n // 2, -3.0, 4.0) + 0.1 * rng.normal(size=n),
            np.sin(t / 3) + 0.01 * t,
            1e-3 * np.cumsum(rng.normal(size=n)),
        ]
    )


def test_each_channel_is_rescaled_to_minus_one_one_before_windowing(monkeypatch):
    seen = _record_training(monkeypatch)
    x = _multichannel()
    w = 16
    TireDetector(window_size=w, domain="TD", random_state=0).fit_predict(x)
    (windows,) = seen
    assert windows.shape == (len(x) - w + 1, x.shape[1] * w)
    for c in range(x.shape[1]):
        expected = tire_utils.ts_to_windows(_rescaled(x[:, c]), 0, w, 1)
        np.testing.assert_allclose(windows[:, c * w : (c + 1) * w], expected)


def test_rescaling_removes_a_positive_affine_map_of_each_channel(monkeypatch):
    seen = _record_training(monkeypatch)
    x = _multichannel()
    det = TireDetector(window_size=16, domain="both", random_state=0)
    det.fit_predict(x)
    det.fit_predict(x * np.array([3.0, 0.01, 7.0, 1e4]) + np.array([5, -2, 0, 1]))
    td, fd, td_mapped, fd_mapped = seen
    np.testing.assert_allclose(td_mapped, td, atol=1e-12)
    np.testing.assert_allclose(fd_mapped, fd, atol=1e-12)


def test_a_constant_channel_is_set_to_zero(monkeypatch):
    seen = _record_training(monkeypatch)
    x = _multichannel()[:, :2]
    x[:, 1] = 7.0
    w, nfft = 16, 30
    TireDetector(window_size=w, nfft=nfft, domain="both", random_state=0).fit_predict(x)
    td, fd = seen
    np.testing.assert_array_equal(td[:, w:], 0.0)
    n_bins = nfft // 2 + 1
    assert np.all(np.isfinite(fd))
    np.testing.assert_array_equal(fd[:, n_bins:], 0.0)
    assert fd[:, :n_bins].min() == pytest.approx(-1.0)
    assert fd[:, :n_bins].max() == pytest.approx(1.0)


@pytest.mark.parametrize("domain", ["TD", "FD", "both"])
def test_a_constant_series_has_no_change_point_even_guided(monkeypatch, domain):
    def no_training(self, *args):
        raise AssertionError("a constant series should not be trained on")

    monkeypatch.setattr(TireDetector, "_train_autoencoder", no_training)
    x = np.column_stack([np.full(200, 3.0), np.full(200, -1.0)])
    det = TireDetector(window_size=16, domain=domain, n_segments=3)
    pred = det.fit_predict(x)
    assert pred.size == 0
    np.testing.assert_array_equal(det.change_scores_, np.zeros(200))


def test_frequency_windows_are_the_dft_of_each_channel(monkeypatch):
    """Eq. 3: one DFT per channel, then concatenation (defaults w=20, nfft=30).

    The DFT of the concatenated windows, cut to ``nfft`` points, saw channel 0
    and ten points of channel 1 only.
    """
    seen = _record_training(monkeypatch)
    x = _multichannel()
    det = TireDetector(domain="FD", random_state=0)
    det.fit_predict(x)
    td, fd = seen
    w, nfft = det.window_size, det.nfft
    expected = np.concatenate(
        [
            tire_utils.calc_fft(td[:, c * w : (c + 1) * w], nfft=nfft)
            for c in range(x.shape[1])
        ],
        axis=1,
    )
    assert fd.shape == (len(x) - w + 1, x.shape[1] * (nfft // 2 + 1))
    np.testing.assert_array_equal(fd, expected)


def test_the_last_channel_reaches_the_frequency_features(monkeypatch):
    """Before eq. 3 was followed, the last channels were cut off by ``nfft``."""
    seen = _record_training(monkeypatch)
    x = _multichannel()
    changed = x.copy()
    changed[150:, 3] = np.sin(np.arange(150))
    det = TireDetector(domain="FD", random_state=0)
    det.fit_predict(x)
    det.fit_predict(changed)
    n_bins = det.nfft // 2 + 1
    before, after = seen[1], seen[3]
    # only the features of the last channel change
    np.testing.assert_array_equal(after[:, : 3 * n_bins], before[:, : 3 * n_bins])
    assert not np.allclose(after[:, 3 * n_bins :], before[:, 3 * n_bins :])


@pytest.mark.parametrize("norm_mode", ["timeseries", "window"])
def test_univariate_frequency_windows_are_the_dft_of_the_windows(
    monkeypatch, norm_mode
):
    seen = _record_training(monkeypatch)
    TireDetector(domain="FD", norm_mode=norm_mode, random_state=0).fit_predict(_steps())
    td, fd = seen
    np.testing.assert_array_equal(
        fd, tire_utils.calc_fft(td, nfft=30, norm_mode=norm_mode)
    )


def _record_distances(monkeypatch):
    calls = []
    original = tire_utils.distance

    def spy(data, window_size):
        calls.append(np.array(data))
        return original(data, window_size)

    monkeypatch.setattr(tire_detector.utils, "distance", spy)
    return calls


@pytest.mark.parametrize("domain", ["TD", "FD"])
def test_representations_are_smoothed_in_every_domain(monkeypatch, domain):
    """Eq. 11: the triangular filter smooths the shared features of TD and FD."""
    rng = np.random.default_rng(0)
    features_td, features_fd = rng.normal(size=(2, 400, 2))
    monkeypatch.setattr(
        TireDetector,
        "_compute_latent_features",
        lambda self, X: (features_td, features_fd),
    )
    calls = _record_distances(monkeypatch)
    w = 20
    TireDetector(window_size=w, domain=domain).fit_predict(_steps())
    (smoothed,) = calls
    features = features_td if domain == "TD" else features_fd
    np.testing.assert_array_equal(smoothed, tire_utils.matched_filter(features, w))


def test_both_domains_weight_then_smooth_the_representations(monkeypatch):
    rng = np.random.default_rng(0)
    features_td, features_fd = rng.normal(size=(2, 400, 2))
    features_fd *= 5
    monkeypatch.setattr(
        TireDetector,
        "_compute_latent_features",
        lambda self, X: (features_td, features_fd),
    )
    calls = _record_distances(monkeypatch)
    w = 20
    TireDetector(window_size=w, domain="both").fit_predict(_steps())
    raw_td, raw_fd, combined = calls
    np.testing.assert_array_equal(raw_td, features_td)
    np.testing.assert_array_equal(raw_fd, features_fd)
    beta = np.quantile(tire_utils.distance(features_td, w), 0.95)
    alpha = np.quantile(tire_utils.distance(features_fd, w), 0.95)
    expected = np.concatenate((features_td * alpha, features_fd * beta), axis=1)
    np.testing.assert_array_equal(combined, tire_utils.matched_filter(expected, w))
