"""Selection of TS-CP2's change points from the change score.

The selection is plain NumPy: these tests run without TensorFlow, unlike
``test_tscp2.py``.
"""

from __future__ import annotations

import numpy as np

from tsseg.algorithms.tscp2.detector import _spaced_peaks

CENTRES = np.arange(0, 100, 5)


def _score(*peaks: tuple[int, float], width: int = 20) -> np.ndarray:
    """Sum of triangular bumps ``(centre, height)`` of half-width ``width``."""
    score = np.zeros(CENTRES.shape[0])
    for centre, height in peaks:
        score += height * np.clip(1.0 - np.abs(CENTRES - centre) / width, 0, None)
    return score


def test_one_centre_per_peak():
    score = _score((30, 1.0), (70, 0.5))
    assert _spaced_peaks(CENTRES, score, 20, n_peaks=2).tolist() == [30, 70]
    below = score > 0.1  # the candidates of the unsupervised mode
    assert _spaced_peaks(CENTRES[below], score[below], 20).tolist() == [30, 70]


def test_centres_exactly_min_distance_apart_are_both_kept():
    score = _score((30, 1.0), (50, 0.9), width=5)
    assert _spaced_peaks(CENTRES, score, 20, n_peaks=2).tolist() == [30, 50]
    assert _spaced_peaks(CENTRES, score, 21, n_peaks=2).tolist() == [0, 30]


def test_ties_go_to_the_earliest_centre():
    score = np.ones(CENTRES.shape[0])
    assert _spaced_peaks(CENTRES, score, 50).tolist() == [0, 50]


def test_closer_centres_complete_the_requested_count():
    score = _score((30, 1.0), (70, 0.5))
    kept = _spaced_peaks(CENTRES, score, 100, n_peaks=3)
    # 30 is closer than 100 points to every other centre: 25 and 35 come next.
    assert kept.tolist() == [25, 30, 35]
    assert _spaced_peaks(CENTRES, score, 20, n_peaks=50).tolist() == CENTRES.tolist()


def test_no_candidate_gives_no_change_point():
    empty = np.array([], dtype=int)
    assert _spaced_peaks(empty, np.array([]), 20).size == 0
    assert _spaced_peaks(empty, np.array([]), 20, n_peaks=2).size == 0
