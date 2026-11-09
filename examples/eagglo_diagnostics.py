"""Measure E-Agglo runtime and diagnose over-segmentation on bundled data.

Run from a source checkout with ``PYTHONPATH=. python
examples/eagglo_diagnostics.py > /tmp/eagglo.jsonl``. Install the ``accelerators``
extra first. Add ``--all-mocap`` for the normalised nine-trial diagnostic.
Timings exclude initial numba compilation. Output is one JSON object
per fit; labels are used only for scoring and the explicitly labelled shuffle
diagnostic, never to choose blocks or penalties. Raw MoCap is used, without the
benchmark's preprocessing, so these are not a replay of its archived scores.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np

from tsseg import load_mocap
from tsseg.algorithms import EAggloDetector
from tsseg.algorithms.utils import extract_cps
from tsseg.metrics import BidirectionalCovering, F1Score


def measure(name, X, truth, *, block=1, penalty=None):
    """Print a fit time, change point count and scores, without tuning."""
    member = None if block == 1 else np.arange(len(X)) // block
    start = perf_counter()
    detector = EAggloDetector(member=member, penalty=penalty).fit(X)
    prediction = detector.predict(X)
    elapsed = perf_counter() - start
    print(
        json.dumps(
            {
                "series": name,
                "n": len(X),
                "block": block,
                "penalty": penalty,
                "seconds": elapsed,
                "n_true": len(truth),
                "n_pred": len(prediction),
                "f1": F1Score(margin=0.01).compute(
                    truth, prediction, n_timepoints=len(X)
                )["score"],
                "bic": BidirectionalCovering().compute(
                    truth, prediction, n_timepoints=len(X)
                )["score"],
            }
        ),
        flush=True,
    )


def shuffle_within_segments(X, truth):
    """Destroy temporal structure using reference boundaries, for diagnosis only."""
    shuffled = X.copy()
    rng = np.random.default_rng(0)
    edges = np.r_[0, truth, len(X)]
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        shuffled[start:end] = X[start:end][rng.permutation(end - start)]
    return shuffled


def main():
    """Separate implementation cost, initial partition and temporal dependence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-mocap", action="store_true")
    args = parser.parse_args()
    # Warm both distance branches and the merge kernels before timing.
    warmup = np.random.default_rng(0).normal(size=(40, 4))
    for member in (None, np.arange(len(warmup)) // 10):
        EAggloDetector(member=member).fit(warmup)

    for n in (1000, 4000):
        X = np.random.default_rng(0).normal(size=(n, 4))
        measure("iid_noise", X, np.array([], dtype=int))

    rng = np.random.default_rng(0)
    X = np.concatenate(
        [rng.normal(0, 1, 100), rng.normal(3, 1, 100), rng.normal(0, 3, 100)]
    ).reshape(-1, 1)
    # Also warm the one-channel distance specialisation.
    EAggloDetector().fit(X[:20])
    for block in (1, 10):
        measure("iid_mean_and_variance", X, np.array([100, 200]), block=block)

    X, labels = load_mocap(trial=0, return_X_y=True)
    truth = extract_cps(labels)
    for block, penalty in ((1, None), (10, None), (50, None), (50, "len_penalty")):
        measure("mocap_86_01_raw", X, truth, block=block, penalty=penalty)

    # Diagnostic only: knowing the true segments allows us to destroy temporal
    # structure while preserving every segment's empirical joint distribution.
    shuffled = shuffle_within_segments(X, truth)
    measure("mocap_86_01_shuffled_within_true_segments", shuffled, truth, block=50)

    if args.all_mocap:
        for trial in range(9):
            X, labels = load_mocap(trial=trial, return_X_y=True)
            truth = extract_cps(labels)
            # Per-channel z-normalisation, as in the benchmark's Normalizer.
            std = X.std(axis=0)
            X = (X - X.mean(axis=0)) / np.where(std == 0, 1, std)
            measure(f"mocap_trial_{trial}_normalised", X, truth, block=50)
            measure(
                f"mocap_trial_{trial}_normalised_shuffled_within_true_segments",
                shuffle_within_segments(X, truth),
                truth,
                block=50,
            )


if __name__ == "__main__":
    main()
