"""Reproduction of the score error of FFT rSVD SST (Weber & Lenz, 2025, Table 4).

Publication: L. Weber and R. Lenz, *Accelerating singular spectrum
transformation for scalable change point detection*, IEEE Access
13:213556-213577, 2025, https://doi.org/10.1109/ACCESS.2025.3640386. The paper
publishes no detection score: its accuracy figure is the mean absolute error
of the score of each fast variant against the SST computed with a full SVD
(Table 4, target rank ``k = 5``, ``k + p = 15``, ``q = 3``): 35.95e-3 for FFT
rSVD SST on its synthetic signals (72,000 score pairs, 30 window sizes from
100 to 5,000) and 1.392e-3 on the UCR archive.

Protocol (Section IV-A): four kinds of change (level, variance, frequency of
a sine, exponential decline), each with a random position and magnitude
("random integer percentage between zero and one"). The signals are generated
here, so no data are downloaded. Their exact form is not published (the
paper's generator lives in a separate experiment repository, not used here):
plateaus and the decline get Gaussian noise of standard deviation 0.01, the
sine frequencies are percentages of 0.1 cycle per point, and the decline
rate a percentage of 0.1 per point. Window sizes 100, 200 and 300 (cost of
the exact SVD), lag ``N // 2``, two draws per kind and size.

Result (2026-11-04): mean absolute error 3.5e-06 over 24 series (4,824
score pairs; about 7 minutes), far below the paper's 35.95e-3, which is used
as the tolerance. The UCR figure is not reproduced (archive not downloaded).
Probable causes of the gap: the approximation error depends on the spectral
decay of the Hankel matrices (Section IV-B), and the signals here, smooth or
with small noise, have a faster decay than the paper's; the paper's figure
also averages window sizes up to 5,000.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsseg.algorithms.sst.sst import sst_scores

pytestmark = pytest.mark.reproduction

PAPER_MAE_SYNTHETIC = 35.95e-3


def _signal(kind: str, N: int, rng: np.random.Generator) -> np.ndarray:
    n = 2 * N - 1 + N // 2 + 200
    c = int(rng.integers(2 * N, n - N // 2))
    t = np.arange(n)
    a, b = rng.integers(1, 100, size=2) / 100
    noise = 0.01 * rng.standard_normal(n)
    if kind == "mean":
        return np.where(t < c, a, b) + noise
    if kind == "variance":
        return np.where(t < c, a, b) * rng.standard_normal(n)
    if kind == "frequency":
        return np.where(
            t < c, np.sin(2 * np.pi * 0.1 * a * t), np.sin(2 * np.pi * 0.1 * b * t)
        )
    return np.where(t < c, 1.0, np.exp(-0.1 * a * (t - c))) + noise


def test_fft_rsvd_score_error_below_paper():
    rng = np.random.default_rng(0)
    errors = []
    for N in (100, 200, 300):
        for kind in ("mean", "variance", "frequency", "decline"):
            for _ in range(2):
                x = _signal(kind, N, rng)[:, None]
                exact = sst_scores(x, N, N // 2, 5, method="exact")
                fast = sst_scores(x, N, N // 2, 5, "fft_rsvd", 10, 3, rng)
                valid = slice(2 * N - 1, len(x) - N // 2 + 1)
                errors.append(np.abs(fast[valid] - exact[valid]))
    mae = float(np.mean(np.concatenate(errors)))
    print(f"FFT rSVD SST mean absolute score error: {mae:.3g}")
    assert mae <= PAPER_MAE_SYNTHETIC
