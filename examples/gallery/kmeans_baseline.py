"""Compare the labelled state examples with k-means, which ignores time order.

The published fits stay fixed. K-means uses the same features and number of
groups; the TICC examples are also refitted without their switching penalty.
"""

import argparse
import json
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import tsseg
from examples.gallery._data import HERE
from tsseg.algorithms import TiccDetector

FIGURES = HERE.parents[1] / "docs/examples/_figures"


def scores(frame, truth, labels):
    return dict(
        ari=float(adjusted_rand_score(frame[truth], labels)),
        transitions=int(np.count_nonzero(np.diff(labels))),
        prediction=np.asarray(labels, dtype=int).tolist(),
    )


def build():
    manifest = json.loads((HERE / "data/manifest.json").read_text())
    audit = {
        "environment": {
            "tsseg": tsseg.__version__,
            **{key: version(key) for key in ["numpy", "scikit-learn"]},
        }
    }
    for slug in ["mocap", "speakers", "sleep"]:
        study = import_module(f"examples.gallery.{slug}").build()
        record = json.loads((FIGURES / f"{slug}.json").read_text())
        if study.source and record["snapshot"] != manifest[study.source]:
            raise ValueError(f"Published fit and snapshot differ: {slug}")
        frame, truth = study.frame, study.truth
        run = record["groups"][0]["runs"][0]
        scaler = StandardScaler()
        X = scaler.fit_transform(frame[record["features"]])
        k = len(set(run["prediction"]))  # same count as the published model
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        check = dict(
            features=record["features"],
            published=dict(
                params=run["params"],
                **scores(frame, truth, np.asarray(run["prediction"])),
            ),
            kmeans=dict(
                params=model.get_params(), **scores(frame, truth, model.fit_predict(X))
            ),
        )
        if run["detector"] == "TiccDetector":
            params = dict(run["params"], beta=0)
            np.random.seed(record["seed"])
            labels = TiccDetector(**params).fit_predict(X)
            check["ticc_no_switch_penalty"] = dict(
                params=params, **scores(frame, truth, labels)
            )
        audit[slug] = check
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FIGURES / "kmeans_baseline.json")
    args = parser.parse_args()
    audit = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(f"Saved the k-means comparison to {args.output}")


if __name__ == "__main__":
    main()
