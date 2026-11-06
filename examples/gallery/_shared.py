"""Runner and audit records shared by the gallery; not part of the tsseg API."""

from __future__ import annotations

import json
import platform
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tsseg
from examples.gallery._data import HERE
from tsseg.algorithms.utils import extract_cps
from tsseg.metrics import AdjustedRandIndex, F1Score, StateMatchingScore

ROOT = HERE.parents[1]
COLORS = ["#147d92", "#dc9251", "#7866ab", "#d65b73", "#4d997b", "#507dbc"]


@dataclass
class Study:
    slug: str
    title: str
    frame: pd.DataFrame
    features: list[str]
    display: list[str]
    detectors: list[tuple[str, object]]
    ylabel: str
    xlabel: str = "Date (UTC)"
    standardize: bool = True
    events: list[tuple] = field(default_factory=list)  # context only, never fitted
    truth: str | None = None
    source: str | None = None
    seed: int = 42
    cp_margin: int = 1


def _coordinates(frame):
    if pd.api.types.is_numeric_dtype(frame.x):
        return frame.x.to_numpy(float), False
    return mdates.date2num(pd.to_datetime(frame.x, utc=True).to_numpy()), True


def _state_order(labels, values):
    """Display only: order arbitrary state IDs by mean first plotted channel."""
    order = sorted(np.unique(labels), key=lambda k: values[labels == k].mean())
    mapping = {int(k): i for i, k in enumerate(order)}
    return np.array([mapping[int(k)] for k in labels]), mapping


def evaluate(study):
    """Fit each group independently and retain every prediction and parameter."""
    groups = (
        study.frame.groupby("group", sort=False)
        if "group" in study.frame
        else [("series", study.frame)]
    )
    results = []
    for name, original in groups:
        frame = original.reset_index(drop=True)
        X = frame[study.features].to_numpy(float)
        if X.ndim != 2 or not np.isfinite(X).all() or len(X) < 2:
            raise ValueError(f"Invalid features in {study.slug}/{name}")
        if study.standardize:
            scaler = StandardScaler().fit(X)
            X = scaler.transform(X)
            scaling = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        else:
            scaling = None
        record = dict(
            group=str(name),
            n=len(X),
            labels=frame.label.astype(str).tolist(),
            x=frame.x.tolist(),
            scaling=scaling,
            runs=[],
        )
        for label, prototype in study.detectors:
            detector = prototype.clone()
            previous_rng = np.random.get_state()
            np.random.seed(study.seed)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    prediction = np.asarray(detector.fit_predict(X))
            finally:
                np.random.set_state(previous_rng)
            kind = detector.get_tag("detector_type")
            run = dict(
                label=label,
                detector=type(detector).__name__,
                params=detector.get_params(deep=False),
                kind=kind,
                prediction=prediction.tolist(),
                warnings=[str(w.message) for w in caught],
            )
            if kind == "change_point_detection":
                if (
                    prediction.ndim != 1
                    or prediction.dtype.kind not in "iu"
                    or np.any(np.diff(prediction) <= 0)
                    or np.any((prediction <= 0) | (prediction >= len(X)))
                ):
                    raise ValueError("CPs must be sorted internal segment starts")
                run["boundaries"] = [str(frame.label.iloc[i]) for i in prediction]
                if study.truth:
                    run["cp_margin"] = study.cp_margin
                    run["f1"] = F1Score(margin=study.cp_margin).compute(
                        extract_cps(frame[study.truth].to_numpy()),
                        prediction,
                        n_timepoints=len(X),
                    )["score"]
            else:
                if prediction.shape != (len(X),) or prediction.dtype.kind not in "iu":
                    raise ValueError("Expected an integer state label per sample")
                ordered, mapping = _state_order(
                    prediction, frame[study.display[0]].to_numpy()
                )
                run["display_states"] = ordered.tolist()
                run["display_mapping"] = mapping
                run["state_profiles"] = [
                    dict(
                        state=int(k),
                        points=int((ordered == k).sum()),
                        means=frame.loc[ordered == k, study.features].mean().to_dict(),
                        display_means=frame.loc[ordered == k, study.display]
                        .mean()
                        .to_dict(),
                    )
                    for k in np.unique(ordered)
                ]
                run["transitions"] = int(np.count_nonzero(np.diff(prediction)))
                if study.truth:
                    truth = frame[study.truth].to_numpy()
                    run["ari"] = AdjustedRandIndex().compute(truth, prediction)["score"]
                    run["sms"] = StateMatchingScore().compute(truth, prediction)[
                        "score"
                    ]
            record["runs"].append(run)
        results.append(record)
    return results


def thumbnail(study, results, output):
    """Compact, legible entry point, with the same primary result as the study."""
    name = results[0]["group"]
    frame = study.frame
    if "group" in frame:
        frame = frame[frame.group == name]
    frame = frame.reset_index(drop=True)
    record = next(g for g in results if g["group"] == name)
    x, _ = _coordinates(frame)
    fig, ax = plt.subplots(figsize=(5, 1.8), layout="constrained")
    for c, col in enumerate(study.display):
        ax.plot(x, frame[col], color=COLORS[c], lw=1)
    primary = record["runs"][0]
    if primary["kind"] == "change_point_detection":
        for cp in primary["prediction"]:
            ax.axvline(x[cp], color="#c4384b", alpha=0.65, lw=1)
    else:
        states = np.asarray(primary["display_states"])
        ends = np.r_[np.flatnonzero(np.diff(states)) + 1, len(states)]
        for start, end in zip(np.r_[0, ends[:-1]], ends, strict=True):
            ax.axvspan(
                x[start],
                x[min(end, len(x) - 1)],
                color=COLORS[states[start]],
                alpha=0.16,
                lw=0,
            )
    ax.set_axis_off()
    ax.margins(x=0)
    fig.savefig(output, metadata={"Date": None, "Creator": "tsseg gallery"})
    plt.close(fig)


def run(study, output=None):
    """Execute one study; write a standalone SVG and complete JSON audit record."""
    from examples.gallery._evidence import summarize, write_evidence
    from examples.gallery._figures import plot

    default_output = output is None
    output = Path(output) if output else ROOT / "docs" / "examples" / "_figures"
    output.mkdir(parents=True, exist_ok=True)
    results = evaluate(study)
    record = dict(
        slug=study.slug,
        title=study.title,
        features=study.features,
        standardized=study.standardize,
        seed=study.seed,
        groups=results,
        environment={
            "python": platform.python_version(),
            # The package's own version: installed metadata can lag an editable install.
            "tsseg": tsseg.__version__,
            **{
                key: version(key)
                for key in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "matplotlib",
                ]
            },
        },
    )
    if study.source:
        record["snapshot"] = json.loads((HERE / "data" / "manifest.json").read_text())[
            study.source
        ]
    for package in ["aeon", "numba"]:
        try:
            record["environment"][package] = version(package)
        except PackageNotFoundError:
            record["environment"][package] = None
    if study.slug == "chloroplast":
        record["annotation_snapshot"] = json.loads(
            (HERE / "data/manifest.json").read_text()
        )["annotations"]
    record["evidence"] = summarize(study, record)
    write_evidence(record["evidence"], output / f"{study.slug}_evidence.txt")
    (output / f"{study.slug}.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    study.frame.to_csv(
        output / f"{study.slug}_features.csv",
        index=False,
        float_format="%.10g",
        lineterminator="\n",
    )
    plot(study, record, output / f"{study.slug}.svg")
    thumbnail_dir = ROOT / "docs" / "_static" / "gallery" if default_output else output
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    thumbnail(study, results, thumbnail_dir / f"{study.slug}_thumb.svg")
    print(
        study.slug,
        [
            (
                g["group"],
                [
                    len(r.get("boundaries", r.get("state_profiles", [])))
                    for r in g["runs"]
                ],
            )
            for g in results
        ],
        flush=True,
    )
    return record
