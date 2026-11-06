"""Figures for the gallery: observations, fitted structure, then a reference."""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from examples.gallery._data import load
from examples.gallery._evidence import intervals, primary
from examples.gallery._shared import COLORS

INK, MUTED, GRID, RED = "#203648", "#617080", "#dce3e7", "#c74d64"
PAPER = "#fcfbf8"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlesize": 11,
            "axes.titlepad": 12,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "svg.fonttype": "none",
            "svg.hashsalt": "tsseg-gallery",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.solid_capstyle": "round",
        }
    )


def canvas(title, subtitle, heights=(3, 2), size=(12.4, 7.6)):
    fig, axes = plt.subplots(
        len(heights), 1, figsize=size, gridspec_kw={"height_ratios": heights}
    )
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.09, top=0.84, hspace=0.65)
    fig.suptitle(title, x=0.09, y=0.965, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.09, 0.915, subtitle, color=MUTED, fontsize=10)
    return fig, np.atleast_1d(axes)


def clean(ax, ylabel=None, dates=False):
    ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=10)
    if dates:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def coordinates(frame):
    return (
        frame.x.to_numpy(float)
        if pd.api.types.is_numeric_dtype(frame.x)
        else mdates.date2num(pd.to_datetime(frame.x))
    )


def change_points(ax, x, cps, color=RED):
    for cp in cps:
        ax.axvline(x[cp], color=color, ls="--", lw=1.4, alpha=0.9)


def state_strip(ax, x, labels, name="Inferred states", y=0, height=0.65):
    edge = x[-1] + np.median(np.diff(x))
    for a, b, k in intervals(labels):
        left, right = x[a], x[b] if b < len(x) else edge
        ax.barh(
            y,
            right - left,
            left=left,
            height=height,
            color=COLORS[k % len(COLORS)],
            edgecolor=PAPER,
            lw=0.3,
        )
        if (right - left) / (edge - x[0]) > 0.065:
            ax.text(
                (left + right) / 2,
                y,
                f"S{k}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
    ax.set_xlim(x[0], edge)
    ax.set_yticks([y], [name])
    ax.spines[["left", "right", "top"]].set_visible(False)


def shade_states(ax, x, labels):
    edge = x[-1] + np.median(np.diff(x))
    for a, b, k in intervals(labels):
        ax.axvspan(
            x[a], x[b] if b < len(x) else edge, color=COLORS[k], alpha=0.09, lw=0
        )


def matrix(ax, evidence, truth_names):
    table = evidence["confusion"]
    counts = np.asarray(table["counts"])
    fractions = counts / np.maximum(counts.sum(axis=0, keepdims=True), 1)
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(counts)):
        for j in range(counts.shape[1]):
            ax.text(
                j,
                i,
                f"{100 * fractions[i, j]:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if fractions[i, j] > 0.6 else INK,
            )
    ax.set_yticks(range(len(counts)), truth_names)
    ax.set_xticks(
        range(counts.shape[1]),
        [f"S{k}\n(n={counts[:, j].sum()})" for j, k in enumerate(table["state_ids"])],
    )
    ax.tick_params(length=0)
    ax.set_title("Composition of each state (columns sum to 100%)")
    ax.spines[:].set_visible(False)


def jekyll(study, record):
    frame = study.frame
    x = coordinates(frame)
    run = primary(record)
    fig, (ax, ref) = canvas(
        study.title,
        "Jekyll & Hyde · 300-word windows · chapter headings excluded from the features",
        heights=(3, 1.2),
        size=(12.4, 6.8),
    )
    for i, (col, name) in enumerate(
        [
            ("first_person", "First person · I, me, my…"),
            ("third_person", "Third person · he, him, his…"),
        ]
    ):
        ax.plot(x, frame[col], color=COLORS[i], lw=1.25, alpha=0.8, label=name)
    change_points(ax, x, run["prediction"])
    ax.legend(loc="upper left", ncol=2)
    clean(ax, "Pronouns / 100 words")
    ax.set_title("A · Pronoun rates; the red dashed line is the PELT change point")
    start = int(frame.reference_start.iloc[0])
    jekyll_start = int(frame.jekyll_start.iloc[0])
    for a, b, color, label in [
        (0, start, "#b4c4cc", "Chapters 1–8\nExternal narrative"),
        (start, jekyll_start, COLORS[0], "Chapter 9\nLanyon"),
        (jekyll_start, x[-1] + 300, COLORS[2], "Chapter 10\nJekyll"),
    ]:
        ref.barh(0, b - a, left=a, height=0.65, color=color)
        ref.text(
            (a + b) / 2,
            0,
            label,
            ha="center",
            va="center",
            color=INK if a == 0 else "white",
            fontsize=9,
        )
    ref.axvline(start, color=INK, ls=":", lw=1.5)
    ref.set_yticks([])
    ref.set_ylim(-0.6, 0.6)
    ref.set_title("B · Reference: chapter structure (not used for fitting)")
    ref.set_xlabel("Word position after removing headings")
    for axis in [ax, ref]:
        axis.set_xlim(0, x[-1] + 300)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}k"))
        clean(axis)
    return fig


def chloroplast(study, record):
    frame = study.frame
    x = coordinates(frame)
    run = primary(record)
    genes = load("annotations")
    fig, (ax, geneax) = canvas(
        study.title,
        "Arabidopsis thaliana chloroplast · NC_000932.1 · 400-base windows · annotations not used for fitting",
        heights=(3, 1.4),
    )
    ax.plot(x, frame.gc, color=COLORS[0], lw=0.85, alpha=0.65)
    change_points(ax, x, run["prediction"])
    ax.set_title("A · GC content; red dashed lines are the PELT change points")
    clean(ax, "GC (%)")
    for a, b in record["evidence"]["high_gc_regions"]:
        for axis in [ax, geneax]:
            axis.axvspan(a, b, color=COLORS[1], alpha=0.13, lw=0)
    for gene in genes.itertuples():
        geneax.barh(
            gene.strand,
            gene.end - gene.x + 1,
            left=gene.x,
            height=0.45,
            color=COLORS[2],
        )
    geneax.set_yticks([-1, 1], ["− strand", "+ strand"])
    geneax.set_ylim(-1.7, 2.2)
    geneax.text(
        104000, 1.65, "16S · 23S · 4.5S", ha="center", fontsize=9, color=COLORS[2]
    )
    geneax.text(
        134000, 0.1, "5S · 4.5S · 23S · 16S", ha="center", fontsize=9, color=COLORS[2]
    )
    evidence = record["evidence"]
    geneax.set_title(
        f"B · Annotated rRNA genes: the regions hold {100 * evidence['rrna_coverage']:.1f}% of rRNA bases; "
        f"{100 * evidence['rrna_fraction_within_regions']:.1f}% of their bases are rRNA"
    )
    geneax.set_xlabel("Reference position (kilobases; the circular origin is fixed)")
    for axis in [ax, geneax]:
        axis.set_xlim(1, 154478)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:g}"))
        clean(axis)
    return fig


def storm_surge(study, record):
    frame = study.frame
    x = coordinates(frame)
    fig, (ax, resid) = canvas(
        study.title,
        "The Battery, New York · hourly NOAA water levels and tide predictions · October–November 2012",
    )
    ax.plot(x, frame.observed, color=COLORS[0], lw=1.15, label="Observed water level")
    ax.plot(
        x,
        frame.predicted,
        color=MUTED,
        lw=0.8,
        alpha=0.65,
        label="Astronomical tide prediction",
    )
    ax.set_title("A · Observed water level and astronomical tide prediction")
    ax.legend(loc="upper left", ncol=2)
    clean(ax, "Water level (m)", True)
    resid.plot(x, frame.residual, color=COLORS[0], lw=1.1)
    resid.axhline(0, color=MUTED, lw=0.7)
    change_points(resid, x, primary(record)["prediction"])
    resid.set_title(
        "B · Residual (observed − predicted); red dashed lines are the PELT change points"
    )
    clean(resid, "Observed − predicted (m)", True)
    for i, event in enumerate(record["evidence"]["events"]):
        t = mdates.date2num(pd.Timestamp(event["peak_time"]))
        resid.annotate(
            event["name"],
            xy=(t, event["peak_m"]),
            xytext=(t - (3 if i == 0 else 8), event["peak_m"] + 0.5),
            fontsize=10,
            fontweight="semibold",
            arrowprops={"arrowstyle": "-", "color": MUTED},
        )
    resid.set_ylim(-0.6, 3.7)
    for axis in [ax, resid]:
        axis.set_xlim(x[0], x[-1])
    return fig


def climate_vocabulary(study, record):
    fig = plt.figure(figsize=(12.4, 7.6))
    gs = fig.add_gridspec(
        2, 2, hspace=0.5, wspace=0.16, left=0.08, right=0.97, bottom=0.08, top=0.83
    )
    fig.suptitle(
        study.title, x=0.08, y=0.965, ha="left", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.08,
        0.915,
        "OpenAlex · titles published 1990–2025 · one PELT fit per expression · "
        "red dashed: change point · black dotted: context, not used for fitting",
        color=MUTED,
    )
    context = {group: (year, label) for group, year, label in study.events}
    for i, group in enumerate(record["groups"]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        frame = study.frame[study.frame.group == group["group"]]
        x = frame.x.to_numpy(float)
        ax.plot(x, frame.rate, color=COLORS[0], lw=1.2, marker="o", ms=2.5)
        change_points(ax, x, group["runs"][0]["prediction"])
        year, label = context[group["group"]]
        ax.axvline(year, color=INK, ls=":", lw=1.2)
        ax.text(
            year - 0.6,
            0.94,
            f"{label}\n({year})",
            transform=ax.get_xaxis_transform(),
            ha="right",
            va="top",
            fontsize=8.5,
            color=MUTED,
        )
        ax.set_title(f"“{group['group']}”")
        clean(ax, study.ylabel if i % 2 == 0 else None)
        ax.set_xlim(1989.5, 2025.5)
        ax.set_ylim(bottom=0)
    return fig


def electricity(study, record):
    frame = study.frame
    x = coordinates(frame)
    z = primary(record)["display_states"]
    fig, (ax, carbon, strip) = canvas(
        study.title,
        "Great Britain · half-hourly generation mix, January 2024 · number of states not given to CLaP",
        heights=(3, 2, 1),
        size=(12.4, 8.6),
    )
    for col, color, label in [("wind", COLORS[0], "Wind"), ("gas", COLORS[1], "Gas")]:
        ax.plot(x, frame[col], color=color, lw=1.1, label=label)
    shade_states(ax, x, z)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("A · Wind and gas shares (nuclear is also fitted, not shown)")
    clean(ax, "Generation share (%)", True)
    carbon.plot(x, frame.carbon_intensity, color=INK, lw=0.9)
    shade_states(carbon, x, z)
    clean(carbon, "gCO₂ / kWh", True)
    carbon.set_title("B · Carbon intensity (not used for fitting)")
    for date, label in [
        ("2024-01-02", "Henk"),
        ("2024-01-21", "Isha"),
        ("2024-01-23", "Jocelyn"),
    ]:
        t = mdates.date2num(pd.Timestamp(date))
        ax.axvline(t, color=MUTED, lw=0.8, ls=":")
        ax.text(
            t,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    state_strip(strip, x, z, "CLaP")
    clean(strip, dates=True)
    strip.set_ylim(-0.6, 0.6)
    strip.set_title("C · CLaP states; dotted lines in A mark named storms")
    for axis in [ax, carbon, strip]:
        axis.set_xlim(x[0], x[-1])
    return fig


def speakers(study, record):
    frame = study.frame
    x = coordinates(frame)
    z = primary(record)["display_states"]
    fig = plt.figure(figsize=(12.4, 8.2))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[2, 1, 2.6],
        hspace=0.75,
        left=0.09,
        right=0.97,
        bottom=0.1,
        top=0.83,
    )
    fig.suptitle(
        study.title, x=0.09, y=0.965, ha="left", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.09,
        0.915,
        "120 distinct spoken-digit clips in six blocks of 20 · TICC told there are three speakers",
        color=MUTED,
    )
    ax, strip, check = [fig.add_subplot(gs[i]) for i in range(3)]
    ax.plot(
        x,
        frame.mfcc_01,
        color=COLORS[0],
        lw=1,
        marker="o",
        ms=2,
        label="MFCC 1 (one of 12 fitted coefficients)",
    )
    shade_states(ax, x, z)
    ax.legend(loc="upper left")
    clean(ax, "Mean MFCC 1")
    ax.set_title("A · First MFCC of each clip")
    state_strip(strip, x, z, "TICC")
    strip.set_ylim(-0.6, 0.6)
    strip.set_xlabel("Utterance index; one observation per clip")
    for start, _, _speaker in intervals(frame.speaker):
        strip.axvline(x[start], color=INK, ls=":", lw=0.8)
    strip.set_title("B · TICC states; dotted lines mark the speaker turns")
    matrix(check, record["evidence"], ["Jackson", "Nicolas", "Theo"])
    ax.set_xlim(x[0], x[-1] + 1)
    return fig


def mocap(study, record):
    frame = study.frame
    x = coordinates(frame)
    z = primary(record)["display_states"]
    fig = plt.figure(figsize=(12.4, 8.6))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[2, 1.25, 2.2],
        hspace=0.7,
        left=0.11,
        right=0.97,
        bottom=0.09,
        top=0.83,
    )
    fig.suptitle(
        study.title, x=0.11, y=0.965, ha="left", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.11,
        0.915,
        "CMU motion capture, subject 86, trial 01 · 120 frames per second · "
        "TICC told there are four activities",
        color=MUTED,
    )
    ax, strip, check = [fig.add_subplot(gs[i]) for i in range(3)]
    for i, (col, name) in enumerate(
        [("right_upper_arm", "Right upper arm"), ("right_thigh", "Right thigh")]
    ):
        ax.plot(x, frame[col], color=COLORS[[0, 2][i]], lw=0.9, label=name)
    shade_states(ax, x, z)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    clean(ax, "Rotation (°)")
    ax.set_title("A · Two of the four fitted joint angles", pad=24)
    state_strip(strip, x, z, "TICC", y=1)
    edge = x[-1] + np.median(np.diff(x))
    for a, b, _ in intervals(frame.activity):
        right = x[b] if b < len(x) else edge
        strip.barh(
            0,
            right - x[a],
            left=x[a],
            height=0.65,
            color="#c9d3d8",
            edgecolor=PAPER,
            lw=1.5,
        )
        strip.text(
            (x[a] + right) / 2,
            0,
            frame.activity_name.iloc[a],
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK,
        )
    strip.set_yticks([0, 1], ["Reference", "TICC"])
    strip.set_ylim(-0.6, 1.6)
    strip.set_xlabel("Time (s)")
    strip.set_title(
        "B · TICC states and the reference activities (not used for fitting)"
    )
    names = frame.groupby("activity").activity_name.first()
    matrix(check, record["evidence"], [names[k] for k in sorted(names.index)])
    for axis in [ax, strip]:
        axis.set_xlim(x[0], edge)
    return fig


def sleep(study, record):
    frame = study.frame
    x = coordinates(frame)
    z = primary(record)["display_states"]
    fig = plt.figure(figsize=(12.4, 10.2))
    gs = fig.add_gridspec(
        4,
        1,
        height_ratios=[2, 1.5, 1, 2.6],
        hspace=0.85,
        left=0.09,
        right=0.97,
        bottom=0.08,
        top=0.84,
    )
    fig.suptitle(
        study.title, x=0.09, y=0.965, ha="left", fontsize=17, fontweight="bold"
    )
    fig.text(
        0.09,
        0.925,
        "Sleep-EDF SC4001 · 23:00–08:00 · 30-second epochs · expert labels not used for fitting",
        color=MUTED,
    )
    ax, expert, strip, check = [fig.add_subplot(gs[i]) for i in range(4)]
    for col, c, label in [
        ("ch0_delta", 0, "Delta · 0.5–4 Hz"),
        ("ch0_sigma", 2, "Sigma · 12–16 Hz"),
    ]:
        ax.plot(x, frame[col], color=COLORS[c], lw=0.8, label=label)
    ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title("A · Two of the seven fitted features", pad=26)
    clean(ax, "Log spectral energy", True)
    # The reference hypnogram uses conventional vertical ordering, not state IDs.
    expert.step(
        x, frame.stage.where(frame.stage != 4, 0.5), where="post", color=INK, lw=1
    )
    expert.set_yticks([0, 0.5, 1, 2, 3], ["Wake", "REM", "N1", "N2", "Deep NREM"])
    expert.invert_yaxis()
    expert.set_title("B · Expert hypnogram (R&K stages 3 and 4 combined)")
    clean(expert, dates=True)
    state_strip(strip, x, z, "HDP-HSMM")
    strip.set_ylim(-0.6, 0.6)
    clean(strip, dates=True)
    strip.set_title("C · HDP-HSMM states")
    matrix(check, record["evidence"], ["Wake", "N1", "N2", "Deep NREM", "REM"])
    for axis in [ax, expert, strip]:
        axis.set_xlim(x[0], x[-1] + 30 / 86400)
        axis.xaxis.set_major_locator(mdates.HourLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    return fig


def plot(study, record, output):
    style()
    builders = {
        "jekyll": jekyll,
        "storm_surge": storm_surge,
        "chloroplast": chloroplast,
        "climate_vocabulary": climate_vocabulary,
        "electricity": electricity,
        "mocap": mocap,
        "speakers": speakers,
        "sleep": sleep,
    }
    figure = builders[study.slug](study, record)
    figure.savefig(output, metadata={"Date": None, "Creator": "tsseg gallery"})
    plt.close(figure)
