"""Run the fixed SNLDS validation protocol against two source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from examples.diagnostics.snlds_audit import implementation_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    script = Path(__file__).with_name("snlds_audit.py").resolve()
    cases = {
        "means": ["--case", "mean-shifts", "--data-seed", "1"],
        "bounce": ["--case", "bounce", "--data-seed", "1"],
        "mocap1": ["--case", "mocap", "--trial", "1"],
        "mocap2": ["--case", "mocap", "--trial", "2"],
    }
    input_files = {}
    for name in ("tssb3", "tssb6", "skab3", "skab6"):
        path = args.data_dir / f"{name}.npz"
        if not path.is_file():
            parser.error(f"Missing prepared series: {path}")
        input_files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases[name] = ["--case", "mocap", "--npz-file", str(path.resolve())]
    fingerprints = {
        "baseline": implementation_fingerprint(args.baseline_root),
        "candidate": implementation_fingerprint(args.candidate_root),
    }
    tasks = []
    for case in cases:
        for seed in (0, 1, 2):
            tasks.append((case, "raw", seed, {}))
            if case not in ("means", "bounce"):
                tasks.append(
                    (case, "moments", seed, {"feature_window": 128, "downsample": 6})
                )
        seeds = (0, 1, 2) if case in ("mocap1", "tssb3") else (0,)
        tasks.extend((case, "baseline", seed, {}) for seed in seeds)
    manifest = {
        "protocol": 1,
        "cases": cases,
        "tasks": tasks,
        "implementations": fingerprints,
        "input_files": input_files,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not args.resume:
            parser.error(
                "Output already exists; use --resume or a new output directory"
            )
        if json.dumps(previous, sort_keys=True) != json.dumps(manifest, sort_keys=True):
            parser.error("Code, inputs or protocol changed; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    def run(task):
        case, variant, seed, params = task
        root = args.baseline_root if variant == "baseline" else args.candidate_root
        expected = fingerprints["baseline" if variant == "baseline" else "candidate"]
        log = args.output / f"{case}-{variant}-{seed}.log"
        if args.resume and log.exists():
            lines = [
                line for line in log.read_text().splitlines() if line.startswith("{")
            ]
            if lines:
                cached = json.loads(lines[-1])
                if cached.get("implementation_sha256") == expected:
                    return {**cached, "case_id": case, "variant": variant}
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(root.resolve()),
            TF_CPP_MIN_LOG_LEVEL="3",
            CUDA_VISIBLE_DEVICES="-1",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        command = [
            args.python,
            str(script),
            *cases[case],
            "--seed",
            str(seed),
            "--params",
            json.dumps(params),
        ]
        with log.open("w") as output:
            completed = subprocess.run(
                command,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=1200,
            )
        lines = [line for line in log.read_text().splitlines() if line.startswith("{")]
        if completed.returncode or not lines:
            raise RuntimeError(f"Failed diagnostic: {log}")
        result = json.loads(lines[-1])
        if result["implementation_sha256"] != expected:
            raise RuntimeError(f"Source changed during validation: {log}")
        return {**result, "case_id": case, "variant": variant}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        with (args.output / "results.jsonl").open("w") as output:
            for future in as_completed(futures):
                result = future.result()
                output.write(json.dumps(result) + "\n")
                output.flush()
                print(
                    result["case_id"],
                    result["variant"],
                    result["seed"],
                    round(result["ari"], 4),
                    round(result["fit_seconds"], 1),
                    flush=True,
                )


if __name__ == "__main__":
    main()
