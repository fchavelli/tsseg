"""Rebuild the small numeric snapshots from the original public sources.

Run from the repository root::

    python -m examples.gallery.prepare --cache /tmp/tsseg-gallery/raw

Only this preparation step needs network access. Audio preparation additionally
requires librosa and pyedflib. Downloads are verified against sources.json;
changed upstream bytes require an explicit manifest update. OpenAlex counts are
revised continuously, so those responses are recorded without a hash check.
Full texts, recordings and raw responses stay in the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokens(text):
    """Join PDF end-of-line hyphenation, then extract lowercase Latin words."""
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)
    return TOKEN.findall(text.lower())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(cache, name, source):
    target = cache / name
    if not target.exists():
        request = Request(source["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=240) as response:
            content = response.read()
        # Never leave an unverified download at the expected cache location.
        if "sha256" not in source:  # Revised upstream (OpenAlex): keep as retrieved.
            target.write_bytes(content)
            return target
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"Upstream bytes changed for {name}; review sources.json")
        target.write_bytes(content)
    if "sha256" in source and digest(target) != source["sha256"]:
        raise ValueError(f"Checksum mismatch: {target}")
    return target


def write(name, frame):
    (HERE / "data").mkdir(exist_ok=True)
    path = HERE / "data" / f"{name}.csv"
    frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return {"file": path.name, "rows": len(frame), "sha256": digest(path)}


def dna(cache):
    seq = "".join((cache / "chloroplast.fasta").read_text().splitlines()[1:]).upper()
    if len(seq) != 154478 or set(seq) - set("ACGT"):
        raise ValueError("Unexpected NC_000932.1 reference sequence")
    rows = []
    # Disjoint windows avoid the apparent precision of overlapping estimates.
    for start in range(0, len(seq) - 399, 400):
        window = seq[start : start + 400]
        rows.append(
            dict(
                x=start + 1,
                label=f"base {start + 1}",
                gc=100 * sum(c in "GC" for c in window) / 400,
                **{c.lower(): window.count(c) / 400 for c in "ACGT"},
            )
        )
    return pd.DataFrame(rows)


def tides(cache):
    observed = pd.DataFrame(json.loads((cache / "water.json").read_text())["data"])
    predicted = pd.DataFrame(
        json.loads((cache / "predictions.json").read_text())["predictions"]
    )
    frame = observed.merge(
        predicted, on="t", suffixes=("_obs", "_pred"), validate="one_to_one"
    )
    frame = frame.rename(columns={"t": "x", "v_obs": "observed", "v_pred": "predicted"})
    expected = pd.date_range("2012-10-15", "2012-11-12 23:00", freq="h")
    if not pd.DatetimeIndex(pd.to_datetime(frame.x)).equals(expected):
        raise ValueError("Observed and predicted tides must cover every expected hour")
    frame["observed"] = pd.to_numeric(frame.observed, errors="coerce")
    frame["predicted"] = pd.to_numeric(frame.predicted, errors="raise")
    if frame.observed.isna().any():
        raise ValueError("Missing observed tide; do not interpolate through an event")
    frame["residual"] = frame.observed - frame.predicted
    frame["label"] = frame.x
    return frame[["x", "label", "observed", "predicted", "residual"]]


CLIMATE_EXPRESSIONS = {
    "climate justice": "openalex_climate_justice.json",
    "loss and damage": "openalex_loss_and_damage.json",
    "just transition": "openalex_just_transition.json",
    "climate emergency": "openalex_climate_emergency.json",
}


def climate_titles(cache):
    """Yearly OpenAlex title counts, relative to titles containing "climate"."""

    def counts(name):
        groups = json.loads((cache / name).read_text())["group_by"]
        return {int(g["key"]): int(g["count"]) for g in groups}

    years = range(1990, 2026)
    climate = counts("openalex_climate.json")
    if any(climate.get(year, 0) == 0 for year in years):
        raise ValueError("Every year needs a non-zero number of climate titles")
    rows = []
    for expression, name in CLIMATE_EXPRESSIONS.items():
        found = counts(name)
        for year in years:
            rate = 1000 * found.get(year, 0) / climate[year]
            rows.append(
                dict(
                    group=expression,
                    x=year,
                    label=str(year),
                    count=found.get(year, 0),
                    climate_titles=climate[year],
                    rate=rate,
                    log_rate=np.log1p(rate),
                )
            )
    return pd.DataFrame(rows)


def main():
    from examples.gallery import prepare_evidence as evidence

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/tsseg-gallery/raw"))
    parser.add_argument(
        "--only",
        choices=[
            "novel",
            "dna",
            "tides",
            "climate_titles",
            "grid",
            "speech",
            "sleep",
            "annotations",
        ],
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    sources = json.loads((HERE / "sources.json").read_text())
    builders = dict(
        novel=evidence.novel,
        dna=dna,
        tides=tides,
        climate_titles=climate_titles,
        grid=evidence.grid,
        speech=evidence.speech,
        sleep=evidence.sleep,
        annotations=evidence.annotations,
    )
    snapshots = (
        json.loads((HERE / "data" / "manifest.json").read_text())
        if (HERE / "data" / "manifest.json").exists()
        else {}
    )
    for name, builder in builders.items():
        if args.only and args.only != name:
            continue
        for filename, source in sources.items():
            if source["study"] == {"annotations": "dna"}.get(name, name):
                fetch(args.cache, filename, source)
        snapshots[name] = write(name, builder(args.cache))
        print(name, snapshots[name], flush=True)
    (HERE / "data" / "manifest.json").write_text(json.dumps(snapshots, indent=2) + "\n")


if __name__ == "__main__":
    main()
