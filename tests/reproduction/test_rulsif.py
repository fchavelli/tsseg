"""RuLSIF (Liu et al., 2013) against the authors' demo and the paper's Table 1."""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms import RuLSIFDetector
from tsseg.algorithms.rulsif.rulsif import rulsif_scores

from ._data import DataFile, fetch

pytestmark = pytest.mark.reproduction

_COMMIT = "496042e79b33c794cf2d6807496499763818508a"

LOGWELL = DataFile(
    url=(
        "https://raw.githubusercontent.com/anewgithubname/change_detection/"
        f"{_COMMIT}/data/logwell.mat"
    ),
    sha256="20993d44b973fd4ed6c667e636e602968bd8be720523cab351df4febba8da746",
    reference=(
        "Well-log data of the demo (demo.m) of the authors' code for Liu, Yamada, "
        "Collier and Sugiyama, Neural Networks 43:72-83, 2013, "
        "github.com/anewgithubname/change_detection"
    ),
    license="MIT (repository of the authors' code)",
    fname="rulsif_logwell.mat",
)

# Peaks of the score in the figure of the authors' demo (demo.png at the
# commit above), read from its pixels (about 3.7 points per pixel) and
# shifted by -55 points: the demo plots the score at the end of the second
# set of subsequences (2n - 2 + k points after its start, 1-based), tsseg at
# the boundary between the two sets (n + (k - 1) // 2 points after it).
_DEMO_PEAKS = np.array([334, 695, 1419, 1541, 1663, 1781])


def test_rulsif_official_demo_logwell():
    """Authors' demo: six peaks near 95 at 334, 695, 1419, 1541, 1663, 1781.

    Reference: the figure produced by ``demo.m`` (the paper reports no
    result on this series), with its settings: alpha = 0.01, n = 50, k = 10,
    5 folds. In the figure the six peaks reach 97-100 % of the maximum of
    the score, the next one (after the second change) 80 %; the score lies
    in [0, (1 - alpha) / alpha] = [0, 99] in theory and its maximum is
    about 95. tsseg (2026-10-09): peaks at 333, 695, 1419, 1540, 1665, 1781,
    heights 94.0-95.7 (the maximum), i.e. within 2 % of each other.

    Tolerance: 8 points on the positions (pixel resolution, plus the random
    folds, whose generator differs from MATLAB's) and heights within
    [0.9, 1] of the maximum, which itself lies in [85, 100].
    """
    from scipy.io import loadmat

    y = loadmat(fetch(LOGWELL))["y"].ravel().astype(float)
    assert y.shape == (2002,)
    detector = RuLSIFDetector(alpha=0.01, n_cps=6)
    cps = detector.fit_predict(y)
    assert np.all(np.abs(cps - _DEMO_PEAKS) <= 8), cps
    scores = detector.scores_
    top = np.nanmax(scores)
    assert 85.0 <= top <= 100.0, top
    assert np.all(scores[cps] >= 0.9 * top), scores[cps]


def _ar2(eps: np.ndarray) -> np.ndarray:
    y = np.zeros(eps.size)
    for t in range(2, eps.size):
        y[t] = 0.6 * y[t - 1] - 0.5 * y[t - 2] + eps[t]
    return y


def _paper_dataset(i: int, rng: np.random.Generator) -> np.ndarray:
    """Datasets 1-4 of Section 4.1: 5,000 points, a change every 100."""
    T, L = 5000, 100
    N = np.arange(T) // L + 1  # segment number, 1-based
    if i == 1:  # jumping mean
        mu = np.zeros(T // L + 1)
        for m in range(2, mu.size):
            mu[m] = mu[m - 1] + m / 16
        return _ar2(mu[N] + 1.5 * rng.standard_normal(T))
    if i == 2:  # scaling variance
        sd = np.where(N % 2 == 1, 1.0, np.log(np.e + N / 4))
        return _ar2(sd * rng.standard_normal(T))
    if i == 3:  # switching covariance
        X = np.empty((T, 2))
        for m in range(1, T // L + 1):
            c = (-1 if m % 2 == 1 else 1) * (4 / 5 + (m - 2) / 500)
            cov = [[1.0, c], [c, 1.0]]
            X[(m - 1) * L : m * L] = rng.multivariate_normal([0, 0], cov, size=L)
        return X
    omega = np.ones(T // L + 1)  # changing frequency
    for m in range(2, omega.size):
        omega[m] = omega[m - 1] * np.log(np.e + m / 2)
    return np.sin(omega[N] * np.arange(1, T + 1)) + 0.8 * rng.standard_normal(T)


def _auc(scores: np.ndarray, cps: np.ndarray, tol: int = 10, gap: int = 20) -> float:
    """ROC area of Section 4.1: alarms = peaks above a threshold, an alarm is
    correct within ``tol`` of a change, alarms closer than ``gap`` to the
    previous one are dropped, TPR = n_cr / n_cp, FPR = (n_al - n_cr) / n_al."""
    v = np.where(np.isnan(scores), -np.inf, scores)
    peaks = np.flatnonzero((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])) + 1
    heights = v[peaks]
    points = [(0.0, 0.0)]
    for eta in np.sort(np.unique(heights))[::-1]:
        alarms = np.sort(peaks[heights >= eta])
        kept = [alarms[0]]
        for a in alarms[1:]:
            if a - kept[-1] >= gap:
                kept.append(a)
        kept = np.asarray(kept)
        hit = np.abs(kept[:, None] - cps[None, :]) <= tol
        n_cr = int(hit.any(axis=0).sum())
        points.append(((kept.size - n_cr) / kept.size, n_cr / cps.size))
    points.append((1.0, 1.0))
    p = np.array(sorted(points))
    return float(np.sum(np.diff(p[:, 0]) * (p[1:, 1] + p[:-1, 1]) / 2))


@pytest.mark.xfail(strict=True, reason="Table 1 of the paper is not reproduced")
@pytest.mark.parametrize(
    "dataset, paper_auc", [(1, 0.848), (2, 0.846), (3, 0.972), (4, 0.844)]
)
def test_rulsif_paper_table1_auc(dataset, paper_auc):
    """Table 1 (RuLSIF, mean over 50 runs, sd 0.012-0.031): 0.848, 0.846, 0.972,
    0.844 on datasets 1-4; tsseg (one run each, 2026-10-09): 0.667, 0.611,
    0.503, 0.396. Not reproduced, hence ``xfail(strict=True)``: a pass would
    signal that the gap has closed.

    Protocol: the datasets and the ROC of Section 4.1 as written, the
    paper's settings (alpha = 0.1, n = 50, k = 10, 5 folds). The estimator
    equals a line-by-line transcription of the authors' MATLAB code (unit
    tests) and reproduces their demo (test above), so the gap most likely
    comes from what the paper leaves unsaid: where the score is placed on
    the time axis (a 10-point tolerance makes the AUC very sensitive to it;
    offsets 0-5 were tried), how the ROC points are integrated with an FPR
    that is not monotone in the threshold, and dataset 4's ``sin(omega x)``,
    whose ``x`` is undefined. Tolerance: three times the largest standard
    deviation of Table 1 (0.093).
    """
    rng = np.random.default_rng(dataset)
    X = _paper_dataset(dataset, rng)
    auc = _auc(rulsif_scores(X), np.arange(100, 5000, 100))
    assert auc >= paper_auc - 0.093, auc
