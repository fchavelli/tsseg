"""Reproduction of Table 1 of the changeforest paper (Londschien et al., 2023).

Publication: M. Londschien, P. Bühlmann and S. Kovács, *Random forests for
change point detection*, JMLR 24(216), 2023, Table 1: average adjusted Rand
index (ARI) of changeforest over 500 simulations, with standard deviations.

Protocol (Section 4.2 of the paper, checked against the paper's simulation
code, github.com/mlondschien/changeforest-simulations, cited in Section 4):
the change in mean (CIM) and change in covariance (CIC) setups (n=600, d=5,
change points at 200 and 400), the Dirichlet setup (n=1000, d=20, ten change
points) and the iris and glass setups, where each class of a classification
data set is one segment: classes with fewer than delta * n observations are
discarded, the observations are shuffled within each class and the classes
concatenated in a random order. ``ChangeForestDetector`` runs unsupervised
with its defaults (those of the package, delta = 0.01, as in the main
simulations). The ARI compares the segmentations as clusterings of the time
points.

Deviations:

- The paper divides each covariate of the class-based setups by the median
  absolute deviation of its consecutive differences (Section 4.2; ``normalize``
  in the simulation code). Random forests are invariant to this per-covariate
  scaling (as the paper notes), so it is skipped.
- The paper's results come from changeforest 0.6.0 (README and
  ``environment.yaml`` of the simulation code; random forests from biosphere
  0.3). This test uses 1.2.1 (biosphere 0.4); according to the package
  changelog, no default has changed since 0.6.0.
- Classes are kept when they have at least delta * n observations, as the
  paper's text says; the simulation code keeps them when they have strictly
  more. No class of iris or glass is at that boundary.

Glass: Table 1 reports d = 8 while the UCI file has 9 covariates. The
simulation code (``load_glass``) reads ``glass.data`` with 10 column names for
its 11 columns, so pandas takes the Id column as index, reads the refractive
index (RI) as "id" and drops it: the paper's glass setup uses the 8 chemical
covariates Na to Fe, and so does this test. With all 9 covariates, the mean
ARI would be 0.964 (sd 0.046), above the published 0.92.

Results (500 simulations, seeds 0-499, changeforest 1.2.1), mean ARI (sd),
tsseg vs Table 1: CIM 0.987 (0.035) vs 0.99 (0.03); CIC 0.920 (0.136) vs
0.93 (0.12); Dirichlet 0.986 (0.018) vs 0.99 (0.02); iris 0.980 (0.044) vs
0.98 (0.04); glass, 8 covariates 0.924 (0.068) vs 0.92 (0.07). Runtime about
4.5 minutes, 3 of them for Dirichlet.

The synthetic twin that runs in CI is ``test_readme_example`` in
``tests/algorithms/test_changeforest.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ._data import fetch

pytestmark = pytest.mark.reproduction

pytest.importorskip("changeforest")
from sklearn.datasets import load_iris  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from tsseg.algorithms import ChangeForestDetector  # noqa: E402

N_SIMULATIONS = 500
DELTA = 0.01

#: Table 1, changeforest row: mean ARI and its standard deviation.
PAPER = {
    "CIM": (0.99, 0.03),
    "CIC": (0.93, 0.12),
    "Dirichlet": (0.99, 0.02),
    "iris": (0.98, 0.04),
    "glass": (0.92, 0.07),
}

DIRICHLET_CPS = [100, 130, 220, 320, 370, 520, 620, 740, 790, 870]


def _labels(cps, n):
    labels = np.zeros(n, dtype=int)
    for cp in cps:
        labels[int(cp) :] += 1
    return labels


def _gaussian(rng, second_segment):
    first = rng.normal(0, 1, (200, 5))
    last = rng.normal(0, 1, (200, 5))
    return np.concatenate([first, second_segment(rng), last]), [200, 400]


def _simulate_cim(rng):
    return _gaussian(rng, lambda r: r.normal(2, 1, (200, 5)))


def _simulate_cic(rng):
    sigma = np.full((5, 5), 0.7)
    np.fill_diagonal(sigma, 1)
    return _gaussian(
        rng, lambda r: r.multivariate_normal(np.zeros(5), sigma, 200, method="cholesky")
    )


def _simulate_dirichlet(rng):
    bounds = [0, *DIRICHLET_CPS, 1000]
    segments = []
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        alpha = rng.uniform(0, 0.2, 20)
        segments.append(rng.dirichlet(alpha, stop - start))
    X = np.concatenate(segments)
    assert np.isfinite(X).all()
    return X, DIRICHLET_CPS


def _class_setup(X, y):
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    kept = classes[counts >= DELTA * n]

    def simulate(rng):
        order = rng.permutation(kept)
        blocks = [X[y == c][rng.permutation(int(np.sum(y == c)))] for c in order]
        cps = np.cumsum([len(b) for b in blocks])[:-1].tolist()
        return np.concatenate(blocks), cps

    return simulate


def _iris():
    data = load_iris()
    return _class_setup(data.data, data.target)


def _glass():
    raw = np.loadtxt(fetch("uci-glass"), delimiter=",")
    # Columns: Id, RI, Na, Mg, Al, Si, K, Ca, Ba, Fe, glass type. RI is left
    # out, as in the paper's simulation code (see the module docstring).
    return _class_setup(raw[:, 2:10], raw[:, 10].astype(int))


SETUPS = {
    "CIM": lambda: _simulate_cim,
    "CIC": lambda: _simulate_cic,
    "Dirichlet": lambda: _simulate_dirichlet,
    "iris": _iris,
    "glass": _glass,
}


@pytest.mark.parametrize("setup", list(SETUPS))
def test_table1_adjusted_rand_index(setup):
    """Mean ARI over 500 simulations within Monte Carlo error of Table 1.

    Tolerance: three standard errors of the difference of two means over 500
    simulations (3 * sqrt(2) * sd / sqrt(500), sd from Table 1), plus 0.005
    for the rounding of the published value to two decimals.
    """
    simulate = SETUPS[setup]()
    scores = []
    for seed in range(N_SIMULATIONS):
        X, true_cps = simulate(np.random.default_rng(seed))
        cps = ChangeForestDetector().fit_predict(X)
        n = len(X)
        scores.append(adjusted_rand_score(_labels(true_cps, n), _labels(cps, n)))
    mean, sd = float(np.mean(scores)), float(np.std(scores))
    paper_mean, paper_sd = PAPER[setup]
    tolerance = 3 * math.sqrt(2) * paper_sd / math.sqrt(N_SIMULATIONS) + 0.005
    print(f"{setup}: ARI {mean:.4f} ({sd:.3f}); paper {paper_mean} ({paper_sd})")
    assert abs(mean - paper_mean) <= tolerance, (setup, mean, sd, tolerance)
