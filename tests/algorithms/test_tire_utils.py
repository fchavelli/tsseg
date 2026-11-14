"""Unit tests of the TIRE helpers: windowing, distance, smoothing and DFT.

The expected values are worked out by hand (or by an explicit DFT sum), so
the tests do not reuse the code they check. The helpers need numpy and scipy
only: these tests run without torch.
"""

import warnings

import numpy as np
import pytest

from tsseg.algorithms.tire import utils


class TestTsToWindows:
    def test_windows_start_at_the_onset_and_move_by_the_stride(self):
        windows = utils.ts_to_windows(np.arange(10.0), 1, 4, 3)
        np.testing.assert_array_equal(windows, [[1, 2, 3, 4], [4, 5, 6, 7]])

    def test_last_window_ends_on_the_last_point(self):
        windows = utils.ts_to_windows(np.arange(8.0), 0, 3, 1)
        assert windows.shape == (6, 3)
        np.testing.assert_array_equal(windows[-1], [5, 6, 7])

    def test_window_normalization_removes_the_mean_of_each_window(self):
        ts = np.array([0.0, 2.0, 4.0, 10.0, 20.0])
        windows = utils.ts_to_windows(ts, 0, 2, 2, normalization="window")
        np.testing.assert_allclose(windows, [[-1, 1], [-3, 3]])


def test_combine_ts_concatenates_the_channels_of_each_window():
    channel_0 = [[0, 1], [2, 3], [4, 5]]
    channel_1 = [[10, 11], [12, 13], [14, 15]]
    combined = utils.combine_ts(np.array([channel_0, channel_1]))
    np.testing.assert_array_equal(
        combined, [[0, 1, 10, 11], [2, 3, 12, 13], [4, 5, 14, 15]]
    )


def test_distance_compares_features_window_size_apart():
    features = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    np.testing.assert_allclose(utils.distance(features, 2), [5.0, 1.0])


def test_new_peak_prominences_scores_every_point_without_warning():
    x = np.array([0.0, 2.0, 1.0, 3.0, 0.0])
    with warnings.catch_warnings():
        # scipy warns about the points that are not peaks (prominence 0)
        warnings.simplefilter("error")
        prominences, left_bases, right_bases = utils.new_peak_prominences(x)
    np.testing.assert_allclose(prominences, [0, 1, 0, 3, 0])
    # the peak at 3 rises from the global minimum on both sides
    assert (left_bases[3], right_bases[3]) == (0, 4)


class TestMatchedFilter:
    def test_impulse_response_is_a_triangle_of_unit_sum(self):
        signal = np.zeros(15)
        signal[7] = 1.0
        out = utils.matched_filter(signal, 3)
        assert out.shape == signal.shape
        np.testing.assert_allclose(out[4:11], np.array([0, 1, 2, 3, 2, 1, 0]) / 9)
        np.testing.assert_allclose(out[:4], 0.0)
        np.testing.assert_allclose(out[11:], 0.0)

    def test_linear_trend_is_kept_away_from_the_edges(self):
        ramp = np.arange(20.0)
        np.testing.assert_allclose(utils.matched_filter(ramp, 3)[3:-3], ramp[3:-3])

    def test_edges_are_padded_with_the_end_values(self):
        out = utils.matched_filter(np.arange(20.0), 3)
        # the three points before the series are copies of the first value:
        # out[0] = (0 * 0 + 1 * 0 + 2 * 0 + 3 * 0 + 2 * 1 + 1 * 2 + 0 * 3) / 9
        assert out[0] == pytest.approx(4 / 9)
        assert out[-1] == pytest.approx(19 - 4 / 9)
        np.testing.assert_allclose(utils.matched_filter(np.full(6, 2.5), 4), 2.5)

    def test_window_size_one_leaves_the_signal_unchanged(self):
        signal = np.random.default_rng(0).normal(size=12)
        np.testing.assert_allclose(utils.matched_filter(signal, 1), signal)

    def test_each_column_is_filtered_on_its_own(self):
        signal = np.random.default_rng(0).normal(size=(30, 3))
        out = utils.matched_filter(signal, 4)
        assert out.shape == signal.shape
        for i in range(signal.shape[1]):
            np.testing.assert_allclose(out[:, i], utils.matched_filter(signal[:, i], 4))


def _dft_magnitudes(x, n):
    """Magnitudes of the n-point DFT of x (zero-padded), by the explicit sum."""
    x = np.concatenate([x, np.zeros(n - len(x))])
    t = np.arange(n)
    return np.array(
        [abs(np.sum(x * np.exp(-2j * np.pi * k * t / n))) for k in range(n)]
    )


class TestCalcFFT:
    # Two windows of a cosine at a quarter of the sampling rate, with offsets
    # 1 and 3: window means 1 and 3, mean of the series 2.
    WINDOWS = np.array([[2.0, 1.0, 0.0, 1.0], [5.0, 3.0, 1.0, 3.0]])

    def test_timeseries_mode_removes_the_series_mean_and_keeps_bin_zero(self):
        # after removing 2: [0, -1, -2, -1] -> |X| = [4, 2, 0] and
        # [3, 1, -1, 1] -> |X| = [4, 4, 0]; then scaled from [0, 4] to [-1, 1]
        out = utils.calc_fft(self.WINDOWS, nfft=4, norm_mode="timeseries")
        np.testing.assert_allclose(out, [[1, 0, -1], [1, 1, -1]], atol=1e-12)

    def test_window_mode_removes_each_window_mean_and_drops_bin_zero(self):
        # after removing 1 and 3: [1, 0, -1, 0] -> |X1, X2| = [2, 0] and
        # [2, 0, -2, 0] -> [4, 0]; then scaled from [0, 4] to [-1, 1]
        out = utils.calc_fft(self.WINDOWS, nfft=4, norm_mode="window")
        np.testing.assert_allclose(out, [[0, -1], [1, -1]], atol=1e-12)

    @pytest.mark.parametrize(
        ("norm_mode", "first_bin"), [("timeseries", 0), ("window", 1)]
    )
    def test_zero_padded_dft_against_the_explicit_sum(self, norm_mode, first_bin):
        rng = np.random.default_rng(0)
        windows = rng.normal(size=(6, 20)) + rng.normal(size=(6, 1))
        nfft = 30
        out = utils.calc_fft(windows, nfft=nfft, norm_mode=norm_mode)
        assert out.shape == (6, nfft // 2 + 1 - first_bin)

        if norm_mode == "timeseries":
            centred = windows - windows.mean()
        else:
            centred = windows - windows.mean(axis=1, keepdims=True)
        magnitudes = np.array([_dft_magnitudes(w, nfft) for w in centred])
        magnitudes = magnitudes[:, first_bin : nfft // 2 + 1]
        lo, hi = magnitudes.min(), magnitudes.max()
        np.testing.assert_allclose(out, 2 * (magnitudes - lo) / (hi - lo) - 1)
        assert out.min() == pytest.approx(-1.0)
        assert out.max() == pytest.approx(1.0)
