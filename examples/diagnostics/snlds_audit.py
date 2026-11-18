"""Small SNLDS audit: run from a checkout with ``python -m examples.diagnostics.snlds_audit``.

Use the same command on main and on the correction branch. This is a diagnostic,
not a benchmark: one seed and two series cannot establish general accuracy.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["mean-shifts", "mocap"], required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    import tensorflow as tf

    from tsseg import load_mocap
    from tsseg.algorithms import SNLDSDetector
    from tsseg.metrics import BidirectionalCovering

    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    if args.case == "mocap":
        X, y = load_mocap(trial=0, return_X_y=True)
    else:
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], [300, 400, 300])
        X = (np.array([0.0, 4.0, -2.0])[y] + rng.normal(0, 0.5, len(y)))[:, None]

    detector = SNLDSDetector(
        n_states=len(np.unique(y)), n_train_steps=args.steps, random_state=args.seed
    )
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
                **vars(args),
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
