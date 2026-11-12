"""Validate a complete SNLDS run and render its JSON artifact and RST table."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["case_id"], record["variant"]].append(record)
    return {
        (case, variant): {
            "runs": len(rows),
            **{
                metric: {
                    "mean": float(np.mean([r[metric] for r in rows])),
                    "std": float(np.std([r[metric] for r in rows])),
                }
                for metric in ("ari", "bic", "sms", "fit_seconds", "predicted_changes")
            },
        }
        for (case, variant), rows in groups.items()
    }


def render_table(groups, cases):
    lines = [
        ".. list-table:: Fixed validation (mean over model seeds)",
        "   :header-rows: 1",
        "",
        "   * - Case",
        "     - Old ARI",
        "     - Raw ARI",
        "     - Moments ARI",
        "     - Old fit (s)",
        "     - Raw fit (s)",
        "     - Moments fit (s)",
    ]
    for case in cases:
        row = []
        for metric in ("ari", "fit_seconds"):
            for variant in ("baseline", "raw", "moments"):
                cell = groups.get((case, variant))
                if cell is None:
                    row.append("n/a")
                elif metric == "ari":
                    value = cell[metric]
                    text = f"{value['mean']:.3f}"
                    if cell["runs"] > 1:
                        text += f" ± {value['std']:.3f}"
                    row.append(text)
                else:
                    row.append(f"{cell[metric]['mean']:.1f}")
        lines.append(f"   * - {case}")
        lines.extend(f"     - {value}" for value in row)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--rst-output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (args.run_dir / "results.jsonl").read_text().splitlines()
    ]
    expected = {(case, variant, seed) for case, variant, seed, _ in manifest["tasks"]}
    actual = [(r["case_id"], r["variant"], r["seed"]) for r in records]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(
            "Incomplete or duplicated run: do not aggregate partial results"
        )
    hashes = defaultdict(set)
    for record in records:
        hashes[record["case_id"]].add(record["input_sha256"])
        name = "baseline" if record["variant"] == "baseline" else "candidate"
        if record["implementation_sha256"] != manifest["implementations"][name]:
            raise ValueError("Mixed implementations in one validation")
        if not all(
            np.isfinite(record[m]) for m in ("ari", "bic", "sms", "fit_seconds")
        ):
            raise ValueError("Non-finite metric")
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("Inputs differ between profiles or seeds")
    groups = summarize(records)
    cases = list(manifest["cases"])
    macro = {}
    for variant in ("baseline", "raw", "moments"):
        selected = [groups[c, variant] for c in cases if (c, variant) in groups]
        macro[variant] = {
            "series": len(selected),
            **{
                metric: float(np.mean([g[metric]["mean"] for g in selected]))
                for metric in ("ari", "bic", "sms", "fit_seconds")
            },
        }
    common = [
        c
        for c in cases
        if all((c, v) in groups for v in ("baseline", "raw", "moments"))
    ]
    macro_real = {
        variant: {
            "series": len(common),
            **{
                metric: float(
                    np.mean([groups[c, variant][metric]["mean"] for c in common])
                )
                for metric in ("ari", "bic", "sms", "fit_seconds")
            },
        }
        for variant in ("baseline", "raw", "moments")
    }
    portable = []
    for record in sorted(
        records, key=lambda r: (r["case_id"], r["variant"], r["seed"])
    ):
        if record.get("npz_file"):
            record = {**record, "npz_file": Path(record["npz_file"]).name}
        portable.append(record)
    artifact = {
        "baseline_commit": "e0b09b3",
        "implementations": manifest["implementations"],
        "input_files": manifest["input_files"],
        "protocol": {
            "development_cases": [
                "means data-seed 0",
                "bounce data-seed 0",
                "mocap0",
                "tssb0",
            ],
            "validation_cases": cases,
            "candidate_seeds": [0, 1, 2],
            "repeated_baseline_cases": ["mocap1", "tssb3"],
            "baseline_other_seeds": [0],
            "candidate_profiles": {
                "raw": {},
                "moments": {"feature_window": 128, "downsample": 6},
            },
            "workers": 4,
            "threads_per_process": 1,
        },
        "limitations": [
            "Eight cases are a diagnostic pilot, not a full benchmark.",
            "No best-profile or best-seed selection per case.",
            "Baseline cells with one run do not estimate seed variance.",
            "Timing combines different defaults and budgets; jobs ran concurrently.",
            "Synthetic bouncing ball is inspired by the task, not the published corpus.",
        ],
        "macro_means": macro,
        "macro_common_real_series": macro_real,
        "results": portable,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.rst_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(artifact, indent=2) + "\n")
    args.rst_output.write_text(render_table(groups, cases))
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
