"""Study-specific checks against references that never enter detector features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from examples.gallery._data import load
from tsseg.algorithms import PeltDetector


def primary(record):
    return record["groups"][0]["runs"][0]


def intervals(labels):
    labels = np.asarray(labels)
    ends = np.r_[np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    return [
        (int(a), int(b), int(labels[a]))
        for a, b in zip(np.r_[0, ends[:-1]], ends, strict=True)
    ]


def confusion(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    true_ids, pred_ids = np.unique(truth), np.unique(prediction)
    counts = np.array(
        [
            [((truth == a) & (prediction == b)).sum() for b in pred_ids]
            for a in true_ids
        ],
        dtype=int,
    )
    rows, cols = linear_sum_assignment(-counts)
    accuracy = float(counts[rows, cols].sum() / len(truth))
    return dict(
        truth_ids=true_ids.tolist(),
        state_ids=pred_ids.tolist(),
        counts=counts.tolist(),
        matched_accuracy=accuracy,
        mapping={
            int(pred_ids[c]): int(true_ids[r]) for r, c in zip(rows, cols, strict=True)
        },
    )


def _onsets(values, penalty, min_size):
    """PELT on one standardised channel, as in the published fit."""
    X = ((values - values.mean()) / values.std())[:, None]
    detector = PeltDetector(
        penalty=penalty, pen_scale="bic", min_size=min_size, jump=1, backend="python"
    )
    return [int(i) for i in detector.fit_predict(X)]


def _jekyll(frame, pred, row, evidence):
    reference = int(frame.reference_start.iloc[0])
    positions = frame.x.iloc[pred].to_numpy()
    nearest = positions[np.argmin(np.abs(positions - reference))]
    row("Start of chapter 9 (Lanyon’s narrative)", f"word {reference:,}")
    row("Detected change point", f"word {nearest:,} (window of 300 words)")
    row(
        "First-person pronouns before / after chapter 9",
        f"{frame.loc[frame.testimony == 0, 'first_person'].mean():.1f} / "
        f"{frame.loc[frame.testimony == 1, 'first_person'].mean():.1f} per 100 words",
    )
    evidence["boundary_error_words"] = int(abs(nearest - reference))


def _chloroplast(frame, pred, row, evidence):
    annotation = load("annotations")
    boundaries = np.r_[0, pred, len(frame)]
    means = [
        float(frame.gc.iloc[a:b].mean())
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    # The descriptive high-GC criterion is independent of gene locations.
    regions = [
        (int(frame.x.iloc[a]), int(frame.x.iloc[b - 1]) + 399)
        for a, b, m in zip(boundaries[:-1], boundaries[1:], means, strict=True)
        if m > 45
    ]
    covered, total = 0, 0
    for gene in annotation.itertuples():
        total += gene.end - gene.x + 1
        covered += sum(
            max(0, min(gene.end, b) - max(gene.x, a) + 1) for a, b in regions
        )
    size = sum(b - a + 1 for a, b in regions)
    evidence.update(
        high_gc_regions=regions,
        rrna_covered_bases=covered,
        rrna_total_bases=total,
        rrna_coverage=covered / total,
        genome_fraction=size / (len(frame) * 400),
        rrna_fraction_within_regions=covered / size if size else 0,
        annotation_snapshot="annotations",
    )
    row(
        "Segments with mean GC above 45%", "; ".join(f"{a:,}–{b:,}" for a, b in regions)
    )
    row(
        "Share of the genome in these segments",
        f"{100 * size / (len(frame) * 400):.1f}%",
    )
    row(
        "Annotated rRNA bases inside them",
        f"{covered:,} of {total:,} ({100 * covered / total:.1f}%)",
    )
    row("Share of their bases annotated as rRNA", f"{100 * covered / size:.1f}%")


def _storm_surge(frame, pred, row, evidence):
    dates = pd.to_datetime(frame.x)
    events = []
    for name, a, b in [
        ("Sandy", "2012-10-28", "2012-11-01"),
        ("November nor’easter", "2012-11-07", "2012-11-09"),
    ]:
        mask = (dates >= a) & (dates < b)
        peak = frame.loc[mask, "residual"].idxmax()
        related = [int(i) for i in pred if mask.iloc[i]]
        events.append(
            dict(
                name=name,
                peak_time=str(frame.x.iloc[peak]),
                peak_m=float(frame.residual.iloc[peak]),
                change_points=related,
            )
        )
        row(
            f"{name}: peak residual",
            f"{frame.residual.iloc[peak]:.2f} m at "
            f"{pd.Timestamp(frame.x.iloc[peak]):%H:%M UTC, %-d %B}; "
            f"{len(related)} change points in the episode",
        )
    evidence["events"] = events


def _electricity(frame, run, row, evidence):
    z = np.asarray(run["display_states"])
    profiles = (
        frame.assign(state=z)
        .groupby("state")[["wind", "gas", "nuclear", "carbon_intensity"]]
        .mean()
    )
    evidence["state_means"] = profiles.to_dict(orient="index")
    for k, p in profiles.iterrows():
        row(
            f"State S{k}: mean shares",
            f"wind {p.wind:.0f}%, gas {p.gas:.0f}%, nuclear {p.nuclear:.0f}%; "
            f"carbon intensity {p.carbon_intensity:.0f} gCO₂/kWh",
        )
    evidence["intervals"] = [
        dict(start=frame.x.iloc[a], end=frame.x.iloc[b - 1], state=k)
        for a, b, k in intervals(z)
    ]
    row(
        "Sequence of states",
        " → ".join(
            f"S{k} from {pd.Timestamp(frame.x.iloc[a]):%-d %B}"
            for a, _, k in intervals(z)
        ),
    )


def _labelled_states(study, frame, run, row, evidence):
    z = np.asarray(run["display_states"])
    table = confusion(frame[study.truth], z)
    evidence["confusion"] = table
    row("Adjusted Rand index with the reference", f"{run['ari']:.3f}")
    row(
        "Accuracy after matching states to reference labels",
        f"{100 * table['matched_accuracy']:.1f}%",
    )
    if study.slug == "sleep":
        counts = np.asarray(table["counts"])
        row("States found / expert stages", f"{len(table['state_ids'])} / 5")
        for label, true_id in [("Deep NREM", 3), ("REM", 4)]:
            r = table["truth_ids"].index(true_id)
            col = int(counts[r].argmax())
            row(
                f"{label}: best-matching state",
                f"S{table['state_ids'][col]} holds "
                f"{100 * counts[r, col] / counts[r].sum():.1f}% of expert {label} "
                f"epochs; {100 * counts[r, col] / counts[:, col].sum():.1f}% of the "
                "state is that stage",
            )
    if study.slug == "mocap":
        counts = np.asarray(table["counts"])
        truth = frame[study.truth].to_numpy()
        reference = np.flatnonzero(np.diff(truth)) + 1
        detected = np.flatnonzero(np.diff(z)) + 1
        row(
            "Transitions found / in the reference",
            f"{len(detected)} / {len(reference)}",
        )
        if len(detected) == len(reference):
            gaps = (
                frame.x.iloc[detected].to_numpy() - frame.x.iloc[reference].to_numpy()
            )
            evidence["transition_gaps_s"] = gaps.round(3).tolist()
            row("Largest gap to a reference transition", f"{np.abs(gaps).max():.2f} s")
        names = frame.groupby(study.truth).activity_name.first()
        for r, true_id in enumerate(table["truth_ids"]):
            col = int(counts[r].argmax())
            row(
                f"{names[true_id].capitalize()}: best-matching state",
                f"S{table['state_ids'][col]} holds "
                f"{100 * counts[r, col] / counts[r].sum():.1f}% of its frames; "
                f"{100 * counts[r, col] / counts[:, col].sum():.1f}% of the state "
                "is that activity",
            )


def _climate_vocabulary(study, record, evidence):
    settings = [(p, m) for p in (1, 2, 4) for m in (2, 3, 5)]
    params = primary(record)["params"]
    rows = []
    for group in record["groups"]:
        name = group["group"]
        frame = study.frame[study.frame.group == name].reset_index(drop=True)
        cps = group["runs"][0]["prediction"]
        years = frame.x.to_numpy()
        onsets = [int(years[i]) for i in cps]
        stable = sum(
            _onsets(frame.log_rate.to_numpy(), p, m) == cps for p, m in settings
        )
        raw = [
            int(years[i])
            for i in _onsets(
                frame.rate.to_numpy(), params["penalty"], params["min_size"]
            )
        ]
        cut = cps[0] if cps else len(frame)
        before, after = frame.rate.iloc[:cut].mean(), frame.rate.iloc[cut:].mean()
        year, event = next(
            ((y, label) for g, y, label in study.events if g == name), (None, "")
        )
        evidence.setdefault("expressions", []).append(
            dict(
                expression=name,
                onsets=onsets,
                same_onsets_in_settings=f"{stable}/{len(settings)}",
                raw_scale_onsets=raw,
                rate_before=float(before),
                rate_after=float(after),
                context_year=year,
                context=event,
            )
        )
        rows.append(
            [
                f"“{name}”",
                ", ".join(map(str, onsets)) or "none",
                f"{stable}/{len(settings)}",
                ", ".join(map(str, raw)) or "none",
                f"{before:.1f} → {after:.1f}",
                f"{event} ({year})",
            ]
        )
    evidence["headers"] = [
        "Expression",
        "Onset (log scale)",
        "Same result",
        "Onset (raw scale)",
        "Rate before → after",
        "Context",
    ]
    evidence["widths"] = "16 12 10 12 14 36"
    evidence["rows"] = rows


def summarize(study, record):
    frame, run = study.frame, primary(record)
    pred = np.asarray(run["prediction"], dtype=int)
    evidence = {"rows": []}

    def row(check, result):
        evidence["rows"].append([check, result])

    if study.slug == "jekyll":
        _jekyll(frame, pred, row, evidence)
    elif study.slug == "chloroplast":
        _chloroplast(frame, pred, row, evidence)
    elif study.slug == "storm_surge":
        _storm_surge(frame, pred, row, evidence)
    elif study.slug == "climate_vocabulary":
        _climate_vocabulary(study, record, evidence)
    elif study.slug == "electricity":
        _electricity(frame, run, row, evidence)
    elif study.slug in ["mocap", "speakers", "sleep"]:
        _labelled_states(study, frame, run, row, evidence)
    return evidence


def write_evidence(evidence, path):
    headers = evidence.get("headers", ["Quantity", "Value"])
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        f"   :widths: {evidence.get('widths', '45 55')}",
        "",
    ]
    for cells in [headers, *evidence["rows"]]:
        lines.append(f"   * - {cells[0]}")
        lines.extend(f"     - {cell}" for cell in cells[1:])
    path.write_text("\n".join(lines) + "\n")
