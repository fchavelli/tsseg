"""Run one guided SNLDS diagnostic with explicit parameters and provenance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score


def implementation_fingerprint(root):
    """Hash executable syntax of the detector package, ignoring docstrings."""
    files = {}
    for path in sorted((root / "tsseg/algorithms/snlds").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                del body[0]
        files[str(path.relative_to(root))] = hashlib.sha256(
            ast.dump(tree, include_attributes=False).encode()
        ).hexdigest()
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["mean-shifts", "bounce", "mocap"], required=True
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--npz-file", type=Path)
    parser.add_argument("--params", type=json.loads, default={})
    args = parser.parse_args()

    import tensorflow as tf

    from tsseg import load_mocap
    from tsseg.algorithms import SNLDSDetector
    from tsseg.metrics import BidirectionalCovering, StateMatchingScore

    source = Path(inspect.getfile(SNLDSDetector)).resolve()
    fingerprint = implementation_fingerprint(source.parents[3])
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    if args.npz_file is not None:
        stored = np.load(args.npz_file)
        X, y = stored["X"], stored["y"]
    elif args.case == "mocap":
        X, y = load_mocap(trial=args.trial, return_X_y=True)
    elif args.case == "bounce":
        rng = np.random.default_rng(args.data_seed)
        speed = rng.uniform(0.08, 0.22)
        phase = (rng.uniform(0.0, 20.0) + np.arange(1600) * speed) % 20.0
        y = (phase >= 10.0).astype(int)
        X = (
            np.where(phase < 10.0, phase, 20.0 - phase) + rng.normal(0.0, 0.1, len(y))
        )[:, None]
    else:
        rng = np.random.default_rng(args.data_seed)
        y = np.repeat([0, 1, 2], [300, 400, 300])
        X = (np.array([0.0, 4.0, -2.0])[y] + rng.normal(0.0, 0.5, len(y)))[:, None]
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    input_hash = hashlib.sha256(X.tobytes() + np.asarray(y).tobytes()).hexdigest()
    params = {"n_states": len(np.unique(y)), "random_state": args.seed, **args.params}
    if args.steps is not None:
        params["n_train_steps"] = args.steps
    detector = SNLDSDetector(**params)
    start = time.perf_counter()
    detector.fit(X)
    fit_seconds = time.perf_counter() - start
    predictions, predict_seconds = [], []
    for _ in range(2):
        start = time.perf_counter()
        predictions.append(detector.predict(X))
        predict_seconds.append(time.perf_counter() - start)
    labels = predictions[0]
    print(
        json.dumps(
            {
                **{
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "parameters": detector.get_params(),
                "implementation_sha256": fingerprint,
                "input_sha256": input_hash,
                "tensorflow": tf.__version__,
                "keras": tf.keras.__version__,
                "n": len(y),
                "d": X.shape[1],
                "n_states": len(np.unique(y)),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "ari": adjusted_rand_score(y, labels),
                "bic": BidirectionalCovering(convert_labels_to_segments=True)(
                    y, labels
                )["score"],
                "sms": StateMatchingScore()(y, labels)["score"],
                "true_changes": int(np.count_nonzero(np.diff(y))),
                "predicted_changes": int(np.count_nonzero(np.diff(labels))),
                "used_states": len(np.unique(labels)),
                "prediction_disagreement": float(
                    np.mean(predictions[0] != predictions[1])
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
