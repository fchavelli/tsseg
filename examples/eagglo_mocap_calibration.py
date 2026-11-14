"""Calibrate and verify a non-guided E-Agglo setting on bundled MoCap trials.

From a source checkout, run::

    PYTHONPATH=. python examples/eagglo_mocap_calibration.py --sweep --verify

Results are written under /tmp/eagglo-mocap-calibration by default. The sweep
uses SciPy distances to share the expensive observation comparisons between
initial partitions, and reuses each merge history for all penalty choices.
Verification fits the public detector without this distance substitution.
Reference labels score predictions and select hyperparameters across trials;
neither labels nor the true change point count are passed to the detector.
The proposed profile is specific to these normalised motion-capture signals,
not a universal default or an independent benchmark of a new method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from tsseg import load_mocap
from tsseg.algorithms import EAggloDetector
from tsseg.algorithms.eagglo import _kernels
from tsseg.algorithms.utils import extract_cps
from tsseg.metrics import BidirectionalCovering, F1Score

# Define the whole exploratory grid before choosing a setting. Log-scaled
# penalties are heuristics for comparison, not a derived information criterion.
ALPHAS = (0.5, 1.0, 2.0)
BLOCKS = (10, 25, 50, 100, 200, 400)
LINEAR_COEFFICIENTS = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
LOG_COEFFICIENTS = (0.5, 1.0, 2.0, 4.0)
KEYS = ["alpha", "block", "family", "coefficient"]


def load_trial(trial):
    """Apply the benchmark's per-channel z-normalisation."""
    X, labels = load_mocap(trial=trial, return_X_y=True)
    std = X.std(axis=0)
    X = (X - X.mean(axis=0)) / np.where(std == 0, 1, std)
    return X, extract_cps(labels)


def score(truth, prediction, n):
    """Score CP lists with the actual series length, without inferred endpoints."""
    return {
        "bic": BidirectionalCovering().compute(truth, prediction, n_timepoints=n)[
            "score"
        ],
        "f1": F1Score(margin=0.01).compute(truth, prediction, n_timepoints=n)["score"],
        "n_true": len(truth),
        "n_pred": len(prediction),
    }


def history_prediction(progression, index, n):
    """Read the interior boundaries of one step, including cyclic mergers."""
    row = progression[index]
    boundaries = np.sort(row[np.isfinite(row)])
    return boundaries[(boundaries > 0) & (boundaries < n)].astype(int)


def sweep(trials, output):
    """Compare 288 settings; labels are used only after each fit."""
    rows = []
    for trial in trials:
        X, truth = load_trial(trial)
        n = len(X)
        for alpha in ALPHAS:
            pair = cdist(X, X)
            np.power(pair, alpha, out=pair)
            distance_scale = float(pair.mean())

            def initialize(data, boundaries, exponent, distances, _pair=pair):
                # Preserve ecp's unscaled initial block distances and operation
                # order. Reduction roundoff can differ from the Numba kernel;
                # --verify recomputes the chosen profile with that kernel.
                starts = boundaries[:-1]
                sizes = np.diff(boundaries)
                m = len(sizes)
                means = np.add.reduceat(
                    np.add.reduceat(_pair, starts, axis=0), starts, axis=1
                )
                means /= sizes[:, None] * sizes[None, :]
                within = np.diag(means).copy()
                distances[:m, :m] = 2 * means.T - within[None, :] - within[:, None]

            with patch.object(_kernels, "_initialize_distances", initialize):
                for block in BLOCKS:
                    detector = EAggloDetector(
                        alpha=alpha, member=np.arange(n) // block
                    ).fit(X)
                    progression, gof = detector.progression, detector.gof_
                    counts = np.arange(len(gof) + 1, 1, -1)
                    penalties = [("mean_diff", -1.0)]
                    penalties += [("linear", c) for c in LINEAR_COEFFICIENTS]
                    penalties += [("log_scaled", c) for c in LOG_COEFFICIENTS]
                    scored = {}
                    for family, coefficient in penalties:
                        if family == "mean_diff":
                            values = [
                                np.diff(np.sort(row[np.isfinite(row)])).mean()
                                for row in progression
                            ]
                            objective = gof + values
                        else:
                            weight = coefficient
                            if family == "log_scaled":
                                weight *= 4 * distance_scale * np.log(n) / block
                            objective = gof - weight * counts
                        index = int(np.argmax(objective))
                        if index not in scored:
                            prediction = history_prediction(progression, index, n)
                            scored[index] = score(truth, prediction, n)
                        rows.append(
                            {
                                "trial": trial,
                                "alpha": alpha,
                                "block": block,
                                "family": family,
                                "coefficient": coefficient,
                                **scored[index],
                            }
                        )
            del initialize, pair
        print(f"Sweep completed: trial {trial}", flush=True)
        pd.DataFrame(rows).to_csv(output / "grid.csv", index=False)

    frame = pd.DataFrame(rows)
    ranked = frame.groupby(KEYS).bic.mean().sort_values(ascending=False, kind="stable")
    ranked.to_csv(output / "ranking.csv")
    held_out = []
    if len(trials) > 1:
        for trial in trials:
            training = frame[frame.trial != trial].groupby(KEYS).bic.mean()
            choice = training.sort_values(ascending=False, kind="stable").index[0]
            test = frame[frame.trial == trial]
            for key, value in zip(KEYS, choice, strict=True):
                test = test[test[key] == value]
            held_out.append(test.iloc[0].to_dict())
    pd.DataFrame(held_out, columns=frame.columns).to_csv(
        output / "leave_one_trial_out.csv", index=False
    )
    print(ranked.head(10).to_string(), flush=True)
    if held_out:
        print(
            "Held-out mean BiCovering:", pd.DataFrame(held_out).bic.mean(), flush=True
        )


def verify(trials, output):
    """Fit the public detector on original data, with no substituted distances."""
    warmup = np.random.default_rng(0).normal(size=(40, 4))
    for alpha, member in ((1.0, None), (0.5, np.arange(40) // 10)):
        EAggloDetector(alpha=alpha, member=member).fit(warmup)
    rows = []
    for trial in trials:
        X, truth = load_trial(trial)
        for setting, alpha, block in (("default", 1.0, 1), ("proposed", 0.5, 200)):
            member = None if block == 1 else np.arange(len(X)) // block
            start = perf_counter()
            prediction = EAggloDetector(alpha=alpha, member=member).fit_predict(X)
            rows.append(
                {
                    "trial": trial,
                    "setting": setting,
                    "alpha": alpha,
                    "block": block,
                    "seconds": perf_counter() - start,
                    **score(truth, prediction, len(X)),
                    "prediction": prediction.tolist(),
                }
            )
            (output / "verified.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(rows[-1]), flush=True)
    print(pd.DataFrame(rows).groupby("setting")[["bic", "f1", "n_pred"]].mean())


def main():
    """Run a reproducible sweep and/or public-detector confirmation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--trials", nargs="+", type=int, choices=range(9), default=range(9)
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/eagglo-mocap-calibration")
    )
    args = parser.parse_args()
    if len(set(args.trials)) != len(args.trials):
        parser.error("--trials must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        sweep(args.trials, args.output)
    if args.verify or not args.sweep:
        verify(args.trials, args.output)


if __name__ == "__main__":
    main()
