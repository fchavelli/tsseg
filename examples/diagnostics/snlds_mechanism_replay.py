"""Replay saved SNLDS weights to inspect continuous uncertainty and context.

Average smoothed discrete marginals over continuous posterior samples. This
approximates integration under q(z|x); it is not exact p(s|x) inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from examples.diagnostics.snlds_audit import implementation_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    import tensorflow as tf

    from tsseg.algorithms import SNLDSDetector
    from tsseg.algorithms.snlds.decoding import viterbi
    from tsseg.algorithms.snlds.vendor import forward_backward_algo

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    directory = args.run_dir
    report = json.loads((directory / "report.json").read_text())
    fingerprint = implementation_fingerprint(Path(__file__).resolve().parents[2])
    if fingerprint != report["implementation_sha256"]:
        parser.error("The fitted implementation differs from the current source")
    with np.load(directory / "trained-arrays.npz") as arrays:
        data = arrays["data"]
        y = arrays["y"][:: report["parameters"]["downsample"]]
        full_z = arrays["z"]
    detector = SNLDSDetector(**report["parameters"])
    width = min(detector.train_window or len(data), len(data))
    model = detector._build_model(data.shape[1], width)
    batch = tf.constant(data[None, :width], tf.float32)
    z = model.inference_network(batch, sample=False)[0][0]
    model.x_emit(z)
    model.z_tran(z)
    model.s_tran(batch)
    names = json.loads((directory / "trained-weights.json").read_text())
    if len(model.weights) != len(names):
        parser.error("The saved weight count differs from the reconstructed model")
    with np.load(directory / "trained-weights.npz") as weights:
        for i, (variable, name) in enumerate(zip(model.weights, names, strict=True)):
            value = weights[f"weight_{i}"]
            if variable.path != name or variable.shape != value.shape:
                parser.error(f"Saved weight path/shape mismatch: {name}")
            variable.assign(value)

    @tf.function(autograph=False)
    def ensemble(batch):
        z = model.inference_network(
            batch, num_samples=args.samples, random_seed=tf.constant([912, 4])
        )[0][:, 0]
        repeated = tf.repeat(batch, args.samples, axis=0)
        log_b, log_a = model.calculate_likelihoods(repeated, z)
        log_gamma = forward_backward_algo.forward_backward(
            log_a, log_b, model.log_init
        )[2]
        return tf.reduce_mean(tf.exp(log_gamma), axis=0), log_a, log_b

    gamma, log_a, log_b = [v.numpy() for v in ensemble(tf.constant(data[None]))]
    mc_labels = gamma.argmax(axis=-1)
    sample_aris = [
        adjusted_rand_score(y, viterbi(a, b, model.log_init.numpy()))
        for a, b in zip(log_a, log_b, strict=True)
    ]

    @tf.function(autograph=False)
    def means(batch):
        return model.inference_network(batch, sample=False)[0][0]

    padded = np.pad(data, ((0, (-len(data)) % width), (0, 0)), mode="edge")
    z_windows = (
        means(tf.constant(padded.reshape(-1, width, data.shape[-1])))
        .numpy()
        .reshape(-1, detector.hidden_dim)[: len(data)]
    )

    @tf.function(autograph=False)
    def score(z):
        b, a = model.calculate_likelihoods(tf.constant(data[None]), z)
        return a, b

    window_a, window_b = score(tf.constant(z_windows[None]))
    window_labels = viterbi(
        window_a.numpy()[0], window_b.numpy()[0], model.log_init.numpy()
    )
    result = {
        "mc_samples": args.samples,
        "mc_ari": adjusted_rand_score(y, mc_labels),
        "sample_ari_mean": float(np.mean(sample_aris)),
        "sample_ari_std": float(np.std(sample_aris)),
        "window_encoded_ari": adjusted_rand_score(y, window_labels),
        "window_latent_mse_vs_full": float(np.mean((z_windows - full_z) ** 2)),
        "weights_verified_by_path_and_shape": True,
    }
    (directory / "replay.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        directory / "replay-arrays.npz",
        mc=mc_labels,
        gamma=gamma,
        window_pred=window_labels,
        z_windows=z_windows,
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
