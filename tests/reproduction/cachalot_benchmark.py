"""Reproduce the Cachalot audit (not a reproduction of the paper's tables).

Run from the repository root with BLAS threads pinned, for example::

    OPENBLAS_NUM_THREADS=1 python -m tests.reproduction.cachalot_benchmark \
        --baseline-ref 8633a8c --output /tmp/cachalot-audit.json

The baseline is loaded from the specified, locally available Git revision.
Only its numerical core runs; detector and metric handling are shared.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import types
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from tsseg import load_mocap
from tsseg.algorithms.cachalot import _core
from tsseg.algorithms.utils import extract_cps
from tsseg.metrics import F1Score


def _prediction(core, y, k=50, nu=0.01):
    candidates = core.lasso_candidate_points(y, k)
    objectives, splits = core.reduced_dynamic_programming(y, candidates)
    return splits[core.select_change_count(objectives, nu)]


def _timings(core):
    rows = []
    for n, k in [(1000, 50), (10000, 50), (10000, 200)]:
        y = np.random.default_rng(42).normal(size=n)
        runs = []
        for _ in range(3):
            start = time.perf_counter()
            candidates = core.lasso_candidate_points(y, k)
            middle = time.perf_counter()
            core.reduced_dynamic_programming(y, candidates)
            end = time.perf_counter()
            runs.append([middle - start, end - middle, end - start])
        rows.append(
            {"n": n, "k": k, "median_seconds": np.median(runs, axis=0).tolist()}
        )
    return rows


def _mocap(core):
    rows = []
    metric = F1Score(margin=0.01)
    for trial in range(9):
        x, labels = load_mocap(trial)
        truth = extract_cps(labels)
        start = time.perf_counter()
        channels = [_prediction(core, channel) for channel in x.T]
        elapsed = time.perf_counter() - start
        union = np.unique(np.concatenate(channels))
        scores = [
            metric.compute(truth, points, n_timepoints=len(x))
            for points in [*channels, union]
        ]
        rows.append(
            {
                "trial": trial,
                "n": len(x),
                "seconds": elapsed,
                "true_cps": truth.tolist(),
                "channel_cps": [points.tolist() for points in channels],
                "union_cps": union.tolist(),
                "scores": scores,
            }
        )
    summary = {
        "union_f1": float(np.mean([r["scores"][-1]["score"] for r in rows])),
        "union_precision": float(np.mean([r["scores"][-1]["precision"] for r in rows])),
        "union_recall": float(np.mean([r["scores"][-1]["recall"] for r in rows])),
        "channel_mean_f1": float(
            np.mean([s["score"] for r in rows for s in r["scores"][:-1]])
        ),
        "oracle_channel_f1": float(
            np.mean([max(s["score"] for s in r["scores"][:-1]) for r in rows])
        ),
    }
    return {"summary": summary, "trials": rows}


def _synthetic(core):
    """Small seeded study of the Section 5.1 model, with explicit deviations.

    We draw 30 independent (layout, level) pairs per K and noise multiplier,
    with 10 noise replicates each. We do not cross 30 layouts with 30 level
    sets as in the paper. Matching uses tsseg F1 with a 5-sample margin.
    """
    rng = np.random.default_rng(7301)
    metric = F1Score(margin=5)
    rows = []
    for k in (5, 10, 15, 20):
        for multiplier in (0.5, 1.5):
            scores = []
            counts = []
            for _ in range(30):
                # Uniform weak compositions, plus 10 points per segment.
                separators = np.sort(
                    rng.choice(500 - 9 * (k + 1) - 1, k, replace=False)
                )
                lengths = np.diff(np.r_[-1, separators, 500 - 9 * (k + 1) - 1]) + 9
                truth = np.cumsum(lengths)[:-1]
                levels = rng.normal(size=k + 1)
                signal = np.repeat(levels, lengths)
                sigma = multiplier * np.min(np.abs(np.diff(levels)))
                for _ in range(10):
                    y = signal + rng.normal(scale=sigma, size=500)
                    predicted = _prediction(core, y)
                    scores.append(metric.compute(truth, predicted, n_timepoints=500))
                    counts.append(len(predicted))
            rows.append(
                {
                    "k": k,
                    "noise_multiplier": multiplier,
                    "replicates": len(scores),
                    "f1": float(np.mean([s["score"] for s in scores])),
                    "precision": float(np.mean([s["precision"] for s in scores])),
                    "recall": float(np.mean([s["recall"] for s in scores])),
                    "mean_predicted_count": float(np.mean(counts)),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="8633a8c")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:tsseg/algorithms/cachalot/_core.py"],
        text=True,
    )
    baseline = types.ModuleType("cachalot_baseline")
    exec(compile(source, f"{args.baseline_ref}/_core.py", "exec"), baseline.__dict__)
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "threadpools": threadpool_info(),
        "baseline_ref": args.baseline_ref,
    }
    for name, core in [("baseline", baseline), ("current", _core)]:
        result[name] = {"timings": _timings(core), "mocap": _mocap(core)}
        print(
            name, result[name]["timings"], result[name]["mocap"]["summary"], flush=True
        )
    result["synthetic"] = _synthetic(_core)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
