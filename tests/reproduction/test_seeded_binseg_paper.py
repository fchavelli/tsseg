"""Reproduction of Table 1 of seeded binary segmentation (Kovács et al., 2023).

Publication: S. Kovács, P. Bühlmann, H. Li and A. Munk, *Seeded binary
segmentation: a general methodology for fast and optimal changepoint
detection*, Biometrika 110(1), 2023; arXiv:2002.06633v1, Table 1: mean (sd)
over 100 simulations of the mean squared error (MSE) of the fitted signal, the
Hausdorff distance, the V measure and the error on the number of change
points, for seeded binary segmentation with greedy (g-SeedBS) and
narrowest-over-threshold (n-SeedBS) selection.

Protocol (Section 4 of the paper): the five test signals of Fryzlewicz (2014,
Appendix B; ``blocks``, ``fms``, ``mix``, ``teeth10``, ``stairs10``) plus
i.i.d. Gaussian noise of the stated standard deviation; seeded intervals with
``m = 2`` (``min_size=1``) and decay ``a = 2^(-1/2)``, or ``2^(-1/8)`` for the
starred rows; model chosen by the strengthened Schwarz information criterion
(the defaults of ``SeededBinSegDetector``). The signals are generated here,
so no data are downloaded. The fitted signal is the mean of each estimated
segment. The V measure compares the segmentations as clusterings of the time
points (``sklearn.metrics.v_measure_score``). The column headed ``N - N̂`` is
compared as ``N̂ - N``: its signs agree with Fryzlewicz (2014, Table 2,
``N̂ - N``) on the WBS rows, e.g. -0.50 for WBS on ``blocks`` here and -0.51
there.

Deviations and unknowns:

- The paper reports Hausdorff distances but not how it scores a segmentation
  without change points; this test reports it (runs with at least one change
  point) without checking it.
- The paper does not give its seeds; this test uses 0-99.
- The paper's total length of the seeded intervals (thousands): blocks 95.3,
  blocks* 329.7, fms 19.1, mix 22.3, teeth10 4.4, stairs10 4.8. Definition 1
  as written gives, without duplicates, 85.5, 316.6, 18.1, 20.7, 4.0, 4.4
  (95.8, 400.6, 19.3, 23.5, 4.7, 5.0 with them; duplicates do not change the
  result). No rounding variant tried matches all six: the authors' intervals
  differ slightly from Definition 1 as written, in a way the paper does not
  state (their code has no license and was not consulted).

Tolerance: three standard errors of a difference of two means of 100 runs,
``3 * sqrt(2) * sd / sqrt(100)``, with the paper's sd, plus the rounding of
the published value.

Results (seeds 0-99, about 50 s), mean MSE / V measure / N̂ - N, tsseg vs
Table 1; all twelve rows within the tolerance:

- blocks g: 3.090 / 0.969 / -0.81 vs 2.922 / 0.970 / -0.61; blocks g*:
  2.747 / 0.973 / -0.64 vs 2.685 / 0.973 / -0.52
- blocks n: 3.102 / 0.968 / -0.90 vs 2.942 / 0.970 / -0.69; blocks n*:
  2.702 / 0.973 / -0.79 vs 2.675 / 0.973 / -0.60
- fms g: 0.005 / 0.959 / 0.12 vs 0.005 / 0.955 / -0.02; fms n: 0.005 /
  0.959 / 0.09 vs 0.004 / 0.958 / -0.04
- mix g: 1.649 / 0.900 / -1.40 vs 1.598 / 0.908 / -1.18; mix n: 1.678 /
  0.898 / -1.52 vs 1.759 / 0.897 / -1.34
- teeth10 g: 0.072 / 0.879 / -1.11 vs 0.061 / 0.933 / -0.19; teeth10 n:
  0.072 / 0.887 / -0.91 vs 0.066 / 0.911 / -0.86
- stairs10 g: 0.024 / 0.981 / 0.47 vs 0.023 / 0.981 / 0.47; stairs10 n:
  0.023 / 0.983 / 0.18 vs 0.021 / 0.984 / 0.10

teeth10 with greedy selection is at the edge of the tolerance (V measure 0.054
below for a tolerance of 0.055, N̂ - N 0.92 below for 1.03) and depends on the
seeds: with the 100 series of one ``default_rng(1)`` stream instead, N̂ - N =
-1.49, and the sSIC keeps no change point at all in 6 of them. On that
stream, the same selection and criterion on random intervals (WBS, 5000
intervals) give N̂ = N in 73 of 100 runs, against 80 in Fryzlewicz (2014,
Table 1), and seeded intervals with ``a = 2^(-1/8)`` in 67: the probable
cause is the interval system (see the total lengths above), not the
selection.

The synthetic twin that runs in CI is ``test_frequent_alternating_changes``
(teeth signal, low noise) and ``test_detects_steps`` in
``tests/algorithms/test_seeded_binseg.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import v_measure_score

from tsseg.algorithms import SeededBinSegDetector

pytestmark = pytest.mark.reproduction

N_SIMULATIONS = 100

# Fryzlewicz (2014), Appendix B: length, change points (last point of each
# segment, 1-based, i.e. the first point of the next one, 0-based), levels,
# noise standard deviation
SIGNALS = {
    "blocks": (
        2048,
        [205, 267, 308, 472, 512, 820, 902, 1332, 1557, 1598, 1659],
        [0, 14.64, -3.66, 7.32, -7.32, 10.98, -4.39, 3.29, 19.03, 7.68, 15.37, 0],
        10.0,
    ),
    "fms": (
        497,
        [139, 226, 243, 300, 309, 333],
        [-0.18, 0.08, 1.07, -0.53, 0.16, -0.69, -0.16],
        0.3,
    ),
    "mix": (
        560,
        [11, 21, 41, 61, 91, 121, 161, 201, 251, 301, 361, 421, 491],
        [7, -7, 6, -6, 5, -5, 4, -4, 3, -3, 2, -2, 1, -1],
        4.0,
    ),
    "teeth10": (140, list(range(11, 132, 10)), [0, 1] * 7, 0.4),
    "stairs10": (150, list(range(11, 142, 10)), list(range(1, 16)), 0.3),
}

# Table 1: (signal, selection, decay) -> MSE, V measure, N^ - N as (mean, sd)
EXPECTED = {
    ("blocks", "greedy", 2**-0.5): ((2.922, 1.077), (0.970, 0.013), (-0.61, 0.803)),
    ("blocks", "greedy", 2**-0.125): ((2.685, 0.828), (0.973, 0.009), (-0.52, 0.577)),
    ("blocks", "narrowest", 2**-0.5): ((2.942, 1.002), (0.970, 0.013), (-0.69, 0.787)),
    ("blocks", "narrowest", 2**-0.125): (
        (2.675, 0.782),
        (0.973, 0.010),
        (-0.60, 0.532),
    ),
    ("fms", "greedy", 2**-0.5): ((0.005, 0.004), (0.955, 0.037), (-0.02, 0.492)),
    ("fms", "narrowest", 2**-0.5): ((0.004, 0.003), (0.958, 0.035), (-0.04, 0.448)),
    ("mix", "greedy", 2**-0.5): ((1.598, 0.517), (0.908, 0.044), (-1.18, 1.048)),
    ("mix", "narrowest", 2**-0.5): ((1.759, 0.605), (0.897, 0.052), (-1.34, 1.199)),
    ("teeth10", "greedy", 2**-0.5): ((0.061, 0.040), (0.933, 0.130), (-0.19, 2.419)),
    ("teeth10", "narrowest", 2**-0.5): (
        (0.066, 0.050),
        (0.911, 0.186),
        (-0.86, 2.903),
    ),
    ("stairs10", "greedy", 2**-0.5): ((0.023, 0.011), (0.981, 0.013), (0.47, 0.745)),
    ("stairs10", "narrowest", 2**-0.5): (
        (0.021, 0.011),
        (0.984, 0.013),
        (0.10, 0.362),
    ),
}


def _signal(name):
    n, cps, levels, sigma = SIGNALS[name]
    bounds = [0, *cps, n]
    f = np.concatenate(
        [np.full(b - a, float(v)) for v, a, b in zip(levels, bounds[:-1], bounds[1:])]
    )
    return f, np.asarray(cps), sigma


def _segment_labels(cps, n):
    labels = np.zeros(n, dtype=int)
    for cp in cps:
        labels[cp:] += 1
    return labels


def _fitted(x, cps):
    bounds = [0, *cps, len(x)]
    return np.concatenate(
        [np.full(b - a, x[a:b].mean()) for a, b in zip(bounds[:-1], bounds[1:])]
    )


def _hausdorff(a, b):
    d = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    return max(d.min(axis=1).max(), d.min(axis=0).max())


def _case_id(case):
    name, selection, decay = case
    return f"{name}-{selection}" + ("*" if decay != 2**-0.5 else "")


CASES = [pytest.param(case, id=_case_id(case)) for case in EXPECTED]


@pytest.mark.parametrize("case", CASES)
def test_table1(case):
    name, selection, decay = case
    f, true_cps, sigma = _signal(name)
    n = len(f)
    truth = _segment_labels(true_cps, n)
    mse, vm, dn, haus = [], [], [], []
    for seed in range(N_SIMULATIONS):
        rng = np.random.default_rng(seed)
        x = f + sigma * rng.standard_normal(n)
        cps = SeededBinSegDetector(selection=selection, decay=decay).fit_predict(x)
        mse.append(np.mean((_fitted(x, cps) - f) ** 2))
        vm.append(v_measure_score(truth, _segment_labels(cps, n)))
        dn.append(len(cps) - len(true_cps))
        if len(cps):
            haus.append(_hausdorff(cps, true_cps))
    obtained = {"MSE": mse, "V": vm, "N^-N": dn}
    report = {
        key: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
        for key, v in obtained.items()
    }
    report["Hausdorff"] = round(float(np.mean(haus)), 2)
    print(_case_id(case), report)  # shown with pytest -s
    failures = []
    for (key, values), (paper_mean, paper_sd), rounding in zip(
        obtained.items(), EXPECTED[case], (0.0005, 0.0005, 0.005)
    ):
        tolerance = 3 * math.sqrt(2) * paper_sd / math.sqrt(N_SIMULATIONS) + rounding
        if abs(np.mean(values) - paper_mean) > tolerance:
            failures.append((key, paper_mean, round(tolerance, 4)))
    assert not failures, (case, report, failures)
