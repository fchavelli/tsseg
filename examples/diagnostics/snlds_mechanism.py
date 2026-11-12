"""Inspect SNLDS representations, regime collapse and decoding on one series.

Labels are used only for evaluation and the guided number of states. Persist
arrays and weights so that a causal inspection need not repeat an expensive fit.
The ablations below are interventions on a fitted model, not new estimators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from examples.diagnostics.snlds_audit import implementation_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["bounce", "mocap", "npz"], required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", type=json.loads, default={})
    parser.add_argument("--skip-cluster-assignment", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case == "npz" and args.input is None:
        parser.error("--case npz requires --input")
    if args.skip_cluster_assignment and args.params.get("init_steps") == 0:
        parser.error("Cluster assignments cannot be removed when initialization is off")

    import tensorflow as tf

    from tsseg import load_mocap
    from tsseg.algorithms import SNLDSDetector
    from tsseg.algorithms.snlds.decoding import viterbi
    from tsseg.algorithms.snlds.vendor import forward_backward_algo

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.case == "npz":
        with np.load(args.input) as stored:
            X, y = stored["X"], stored["y"]
    elif args.case == "mocap":
        X, y = load_mocap(trial=args.trial, return_X_y=True)
    else:
        rng = np.random.default_rng(1)
        speed = rng.uniform(0.08, 0.22)
        phase = (rng.uniform(0.0, 20.0) + np.arange(1600) * speed) % 20.0
        y = (phase >= 10.0).astype(int)
        X = (
            np.where(phase < 10.0, phase, 20.0 - phase) + rng.normal(0.0, 0.1, len(y))
        )[:, None]
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    K = len(np.unique(y))
    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = {}

    def inspect_model(detector, data, stage):
        model = detector._model
        target = y[:: detector.downsample]

        @tf.function(autograph=False, reduce_retracing=True)
        def inspect_batch(batch, sample):
            z, entropy, _ = model.inference_network(
                batch, random_seed=tf.constant([923, 7]), sample=sample
            )
            z = z[0]
            emission = model.x_emit(z)
            log_x = emission.log_prob(batch)
            log_b, log_a = model.calculate_likelihoods(batch, z)
            _, _, log_gamma, _ = forward_backward_algo.forward_backward(
                log_a, log_b, model.log_init
            )
            return (
                z,
                emission.mean(),
                log_x,
                log_b,
                log_a,
                log_gamma,
                model.z_tran(z[:, :-1]).mean(),
                entropy[0],
            )

        @tf.function(autograph=False)
        def evaluate(batch):
            return model(batch, num_samples=4, random_seed=tf.constant([441, 7]))[
                "elbo"
            ] / tf.cast(tf.shape(batch)[1], tf.float32)

        batch = tf.constant(data[None], tf.float32)
        values = [v.numpy()[0] for v in inspect_batch(batch, False)]
        z, reconstruction, log_x, log_b, log_a, log_gamma, dynamics, entropy = values
        log_init = model.log_init.numpy()
        predicted = viterbi(log_a, log_b, log_init)
        # Removing a state-independent emission term must preserve decoding.
        log_dynamics = log_b - log_x[:, None]
        dynamics_only = viterbi(log_a, log_dynamics, log_init)
        transition_only = viterbi(log_a, np.zeros_like(log_b), log_init)
        uniform_transition = viterbi(
            np.full_like(log_a, -np.log(K)), log_dynamics, log_init
        )
        gamma = np.exp(log_gamma)
        latent_clusters = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(z)
        observed_clusters = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(
            data
        )
        width = min(detector.train_window or len(data), len(data))
        starts = np.random.default_rng(441).integers(0, len(data) - width + 1, 16)
        windows = data[starts[:, None] + np.arange(width)[None]]
        # Fixed observations and random seed at both stages; a Monte Carlo
        # training-context ELBO estimate, not held-out predictive performance.
        elbo = float(evaluate(tf.constant(windows, tf.float32)))
        sampled = [v.numpy()[0] for v in inspect_batch(batch, True)]
        sample_labels = viterbi(sampled[4], sampled[3], log_init)
        scale = model.z_tran.cov_mat.numpy()
        spread = np.sqrt(np.mean(np.var(dynamics, axis=1) / np.square(scale)))
        summary = {
            "ari": adjusted_rand_score(target, predicted),
            "latent_kmeans_ari": adjusted_rand_score(target, latent_clusters),
            "observed_kmeans_ari": adjusted_rand_score(target, observed_clusters),
            "transition_only_ari": adjusted_rand_score(target, transition_only),
            "uniform_transition_ari": adjusted_rand_score(target, uniform_transition),
            "sampled_z_ari": adjusted_rand_score(target, sample_labels),
            "emission_removal_disagreement": float(np.mean(predicted != dynamics_only)),
            "used_states": len(np.unique(predicted)),
            "changes": int(np.count_nonzero(np.diff(predicted))),
            "occupancy": np.bincount(predicted, minlength=K).tolist(),
            "mean_posterior_occupancy": gamma.mean(axis=0).tolist(),
            "posterior_entropy": float(-np.mean(np.sum(gamma * log_gamma, axis=1))),
            "reconstruction_mse": float(np.mean((data - reconstruction) ** 2)),
            "window_elbo_per_point": elbo,
            "mean_log_emission": float(log_x.mean()),
            "continuous_entropy": float(entropy.mean()),
            "dynamics_spread_over_noise": float(spread),
            "latent_std": z.std(axis=0).tolist(),
            "dynamics_scale": scale.tolist(),
            "emission_scale": model.x_emit.cov_mat.numpy().tolist(),
            "confusion": confusion_matrix(target, predicted).tolist(),
        }
        if X.shape[1] == 1:
            summary["velocity_sign_ari"] = adjusted_rand_score(
                predicted, np.gradient(X[:: detector.downsample, 0]) > 0
            )
        arrays = dict(
            X=X,
            y=y,
            data=data,
            z=z,
            reconstruction=reconstruction,
            log_b=log_b,
            log_a=log_a,
            log_gamma=log_gamma,
            dynamics=dynamics,
            log_init=log_init,
            predicted=predicted,
            latent_clusters=latent_clusters,
            observed_clusters=observed_clusters,
        )
        np.savez_compressed(args.output / f"{stage}-arrays.npz", **arrays)
        np.savez_compressed(
            args.output / f"{stage}-weights.npz",
            **{f"weight_{i}": v.numpy() for i, v in enumerate(model.weights)},
        )
        (args.output / f"{stage}-weights.json").write_text(
            json.dumps([v.path for v in model.weights], indent=2) + "\n"
        )
        snapshots[stage] = summary
        (args.output / "snapshots.json").write_text(
            json.dumps(snapshots, indent=2) + "\n"
        )
        print(json.dumps({"stage": stage, **summary}), flush=True)

    class ProbeDetector(SNLDSDetector):
        def _initialize(self, data, next_batch, seed):
            if args.skip_cluster_assignment:
                # Keep exactly the reconstruction updates and training-window
                # RNG consumption of the control. Undo only the subsequent
                # assignments to dynamics, initial means and transition logits.
                model = self._model
                model.z_tran(tf.zeros([1, 1, self.hidden_dim]))
                model.s_tran(tf.zeros([1, 1, data.shape[1]]))
                variables = (
                    list(model.z_tran.weights)
                    + list(model.s_tran.weights)
                    + list(model.z0_dist.weights)
                )
                original = [v.numpy().copy() for v in variables]
            super()._initialize(data, next_batch, seed)
            if args.skip_cluster_assignment:
                for variable, value in zip(variables, original, strict=True):
                    variable.assign(value)
            inspect_model(self, data, "initialized")

    detector = ProbeDetector(n_states=K, random_state=args.seed, **args.params)
    start = time.perf_counter()
    detector.fit(X)
    fit_seconds = time.perf_counter() - start
    data = detector._features(X)[:: detector.downsample]
    data = ((data - detector._mean) / detector._std).astype(np.float32)
    inspect_model(detector, data, "trained")
    report = {
        "case": args.case,
        "trial": args.trial,
        "input": str(args.input),
        "skip_cluster_assignment": args.skip_cluster_assignment,
        "input_sha256": hashlib.sha256(X.tobytes() + y.tobytes()).hexdigest(),
        "implementation_sha256": implementation_fingerprint(
            Path(__file__).resolve().parents[2]
        ),
        "parameters": detector.get_params(),
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "fit_seconds_including_initial_probe": fit_seconds,
        "true_changes": int(np.count_nonzero(np.diff(y))),
        "snapshots": snapshots,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
