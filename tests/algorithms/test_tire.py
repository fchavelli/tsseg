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
