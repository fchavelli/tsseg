"""Figures of the evaluation page.

Sphinx extension providing the ``tsseg-metric-figure`` directive: an inline SVG
that illustrates one metric of :mod:`tsseg.metrics` on a small example. The
scores printed in the figures are computed by the metrics themselves at build
time, so they follow the code; the build fails when a hand-drawn quantity (an
IoU, a reward, an error type) disagrees with the metric.

The SVG carries classes only (``tm-*``); its colours come from
``_static/css/tsseg.css`` and follow the light and dark themes.
"""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

# Change point example, shared by the four change point figures
N = 1000
TRUE_CPS = [300, 600, 800]
PRED_CPS = [320, 450, 790]
MARGIN = 50  # F1Score(margin=50): 50 points on each side
SIGMA_FRACTION = 0.025  # GaussianF1Score(sigma_fraction=0.025): sigma = 25 points

# State example (A, B, C = 0, 1, 2)
STATE_TRUE = [(0, 300, 0), (300, 600, 1), (600, 800, 0), (800, 1000, 2)]
STATE_NAMES = "ABC"

W = 720  # viewBox width
LABEL_X = 8  # row labels
X0 = 100  # left end of the time axis


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


class Fig:
    """A tiny SVG builder on a time axis of ``n`` points."""

    def __init__(self, height: int, x1: int = 700, n: int = N):
        self.height = height
        self.x1 = x1
        self.n = n
        self.parts: list[str] = []

    def x(self, t: float) -> float:
        return round(X0 + (self.x1 - X0) * t / self.n, 2)

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, cls="tm-muted", anchor="start", title=None):
        tip = f"<title>{_esc(title)}</title>" if title else ""
        self.add(
            f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">'
            f"{tip}{_esc(s)}</text>"
        )

    def line(self, x1, y1, x2, y2, cls, title=None):
        tip = f"<title>{_esc(title)}</title>" if title else ""
        self.add(
            f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">{tip}</line>'
        )

    def rect(self, x, y, w, h, cls, rx=3, title=None):
        tip = f"<title>{_esc(title)}</title>" if title else ""
        self.add(
            f'<rect class="{cls}" x="{x}" y="{y}" width="{max(w, 0)}" height="{h}" '
            f'rx="{rx}">{tip}</rect>'
        )

    def circle(self, x, y, r, cls, title=None):
        tip = f"<title>{_esc(title)}</title>" if title else ""
        self.add(f'<circle class="{cls}" cx="{x}" cy="{y}" r="{r}">{tip}</circle>')

    def row_label(self, y, s):
        self.text(LABEL_X, y, s, cls="tm-row")

    def baseline(self, y):
        self.line(self.x(0), y, self.x(self.n), y, "tm-base")

    def axis_ends(self, y):
        self.text(self.x(0), y, "0", anchor="middle")
        self.text(self.x(self.n), y, f"n = {self.n:,}", anchor="end")

    def states(self, y, h, segments, names=None, title=""):
        """A strip of state segments with a 2px gap between them."""
        for a, b, s in segments:
            x0, x1 = self.x(a) + 1, self.x(b) - 1
            name = names[s] if names else str(s)
            self.rect(
                x0,
                y,
                x1 - x0,
                h,
                f"tm-s{s % 3}",
                rx=4,
                title=f"{title}[{a}, {b}): {name}",
            )
            if x1 - x0 >= 12:
                self.text(
                    (x0 + x1) / 2,
                    y + h / 2 + 4,
                    name,
                    cls="tm-state-txt",
                    anchor="middle",
                )

    def svg(self, label: str) -> str:
        return (
            f'<svg class="tm-svg" viewBox="0 0 {W} {self.height}" role="img" '
            f'aria-label="{_esc(label)}">'
            f"<title>{_esc(label)}</title>{''.join(self.parts)}</svg>"
        )


def _labels(segments, n=N) -> np.ndarray:
    y = np.empty(n, dtype=int)
    for a, b, s in segments:
        y[a:b] = s
    return y


def _check(name: str, ours: float, theirs: float) -> None:
    if not math.isclose(ours, theirs, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(
            f"metric figure: {name} = {ours}, but the metric gives {theirs}"
        )


# ------------------------------------------------------------- change points
def _cp_rows(f: Fig, y_true: int, y_pred: int, h: int = 26):
    """Truth ticks above, predicted ticks below, with their values."""
    f.row_label(y_true + h / 2 + 4, "Truth")
    f.row_label(y_pred + h / 2 + 4, "Prediction")
    f.baseline(y_true + h / 2)
    f.baseline(y_pred + h / 2)
    for t in TRUE_CPS:
        f.text(f.x(t), y_true - 6, t, anchor="middle")
    for t in PRED_CPS:
        f.text(f.x(t), y_pred + h + 15, t, anchor="middle")


def fig_f1() -> str:
    from tsseg.metrics import F1Score

    res = F1Score(margin=MARGIN).compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
    f = Fig(176)
    yt, yp, h = 24, 76, 26
    # tolerance windows
    for i, t in enumerate(TRUE_CPS):
        x0, x1 = f.x(t - MARGIN), f.x(t + MARGIN)
        f.rect(
            x0,
            yt - 2,
            x1 - x0,
            yp + h - yt + 4,
            "tm-window",
            rx=6,
            title=f"window [{t - MARGIN}, {t + MARGIN}] around {t}",
        )
        if i == 0:
            f.text(x0 - 5, (yt + h + yp) / 2 + 4, f"±{MARGIN}", anchor="end")
    _cp_rows(f, yt, yp, h)
    # matching, as F1Score does it: closest free prediction within the margin
    free, matched_true, matched_pred = set(PRED_CPS), set(), set()
    for t in sorted(TRUE_CPS):
        close = [p for p in free if abs(t - p) <= MARGIN]
        if close:
            p = min(close, key=lambda q: abs(t - q))
            free.remove(p)
            matched_true.add(t)
            matched_pred.add(p)
    for t in TRUE_CPS:
        ok = t in matched_true
        f.line(
            f.x(t),
            yt,
            f.x(t),
            yt + h,
            "tm-true" if ok else "tm-true is-missed",
            title=f"true change point {t}" + ("" if ok else ": missed (FN)"),
        )
        if not ok:
            f.text(f.x(t) + 7, yt + h / 2 + 4, "FN", cls="tm-ink")
    for p in PRED_CPS:
        ok = p in matched_pred
        f.line(f.x(p), yp, f.x(p), yp + h, "tm-pred" if ok else "tm-pred is-bad")
        f.circle(
            f.x(p),
            yp,
            4,
            "tm-dot" if ok else "tm-dot is-bad",
            title=f"predicted {p}: " + ("TP" if ok else "FP"),
        )
        f.text(f.x(p) + 7, yp + h / 2 + 4, "TP" if ok else "FP", cls="tm-ink")
    tp = len(matched_pred)
    fp, fn = len(PRED_CPS) - tp, len(TRUE_CPS) - tp
    _check("F1 precision", tp / len(PRED_CPS), res["precision"])
    _check("F1 recall", tp / len(TRUE_CPS), res["recall"])
    f.text(
        LABEL_X,
        166,
        f"{tp} TP, {fp} FP, {fn} FN · precision {tp}/{len(PRED_CPS)} = "
        f"{_fmt(res['precision'])} · recall {tp}/{len(TRUE_CPS)} = "
        f"{_fmt(res['recall'])} · F1 = {_fmt(res['score'])}",
        cls="tm-score",
    )
    return f.svg(
        f"F1Score with a margin of {MARGIN} points: true change points "
        f"{TRUE_CPS}, predicted {PRED_CPS}, F1 {_fmt(res['score'])}"
    )


def fig_gaussian_f1() -> str:
    from tsseg.metrics import GaussianF1Score

    res = GaussianF1Score(sigma_fraction=SIGMA_FRACTION).compute(
        TRUE_CPS, PRED_CPS, n_timepoints=N
    )
    sigma = SIGMA_FRACTION * N
    f = Fig(186)
    y_top, base, amp = 22, 124, 78  # truth ticks, baseline of the rewards, height
    f.row_label(y_top + 12, "Truth")
    f.row_label(base + 4, "Prediction")
    f.baseline(base)
    f.baseline(y_top + 8)
    for t in TRUE_CPS:
        f.text(f.x(t), y_top - 6, t, anchor="middle")
        f.line(f.x(t), y_top, f.x(t), y_top + 16, "tm-true")
        f.line(f.x(t), y_top + 16, f.x(t), base, "tm-guide")
        pts = []
        for k in range(-40, 41):
            u = t + k * 4 * sigma / 40
            g = math.exp(-((u - t) ** 2) / (2 * sigma**2))
            pts.append(f"{f.x(u)},{round(base - amp * g, 2)}")
        f.add(f'<polyline class="tm-gauss" points="{" ".join(pts)}"/>')
    # reward of each prediction: greedy matching on the highest reward first
    pairs = sorted(
        (
            (math.exp(-((p - t) ** 2) / (2 * sigma**2)), t, p)
            for t in TRUE_CPS
            for p in PRED_CPS
        ),
        reverse=True,
    )
    used_t, reward = set(), {p: 0.0 for p in PRED_CPS}
    for r, t, p in pairs:
        if r > 0 and t not in used_t and reward[p] == 0.0:
            used_t.add(t)
            reward[p] = r
    total = sum(reward.values())
    _check("Gaussian F1 matched weight", total, res["matched_weight"])
    for p in PRED_CPS:
        r = reward[p]
        y = base - amp * r
        ok = r > 1e-3
        f.line(f.x(p), base, f.x(p), y, "tm-pred" if ok else "tm-pred is-bad")
        f.circle(
            f.x(p),
            y,
            4,
            "tm-dot" if ok else "tm-dot is-bad",
            title=f"predicted {p}: reward {r:.3f}",
        )
        t = min(TRUE_CPS, key=lambda c: abs(c - p))
        left = ok and p < t
        f.text(
            f.x(p) + (-7 if left else 7),
            y - 4 if ok else base - 6,
            _fmt(r),
            cls="tm-ink",
            anchor="end" if left else "start",
        )
        f.text(f.x(p), base + 18, p, anchor="middle")
    k = len(PRED_CPS)
    f.text(
        LABEL_X,
        176,
        f"W = {' + '.join(_fmt(reward[p]) for p in PRED_CPS)} = {_fmt(total)} · "
        f"precision W/{k} = {_fmt(res['precision'])} · recall "
        f"W/{len(TRUE_CPS)} = {_fmt(res['recall'])} · Gaussian F1 = "
        f"{_fmt(res['score'])}",
        cls="tm-score",
    )
    f.text(
        f.x(TRUE_CPS[0] - 3 * sigma) - 6,
        base - 26,
        f"σ = {sigma:g} points",
        anchor="end",
    )
    return f.svg(
        f"GaussianF1Score with sigma = {sigma:g} points: each prediction earns the "
        f"height of the Gaussian around the nearest free true change point; "
        f"Gaussian F1 {_fmt(res['score'])}"
    )


def fig_hausdorff() -> str:
    from tsseg.metrics import HausdorffDistance

    res = HausdorffDistance().compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
    f = Fig(180)
    yt, yp, h = 24, 96, 26
    _cp_rows(f, yt, yp, h)
    pairs = set()
    for t in TRUE_CPS:
        pairs.add((t, min(PRED_CPS, key=lambda p: abs(p - t))))
    for p in PRED_CPS:
        pairs.add((min(TRUE_CPS, key=lambda t: (abs(p - t), -t)), p))
    worst = max(abs(t - p) for t, p in pairs)
    _check("Hausdorff distance", worst, res["score"])
    for t, p in sorted(pairs):
        d = abs(t - p)
        cls = "tm-link is-max" if d == worst else "tm-link"
        f.line(f.x(t), yt + h, f.x(p), yp, cls, title=f"{t} to {p}: {d} points")
        mx, my = (f.x(t) + f.x(p)) / 2, (yt + h + yp) / 2
        f.text(mx + 8, my + 4, d, cls="tm-ink-strong" if d == worst else "tm-ink")
    for t in TRUE_CPS:
        f.line(f.x(t), yt, f.x(t), yt + h, "tm-true")
    for p in PRED_CPS:
        f.line(f.x(p), yp, f.x(p), yp + h, "tm-pred")
    f.text(
        LABEL_X,
        170,
        f"largest distance to the nearest change point of the other set: "
        f"Hausdorff = {res['score']:g} points",
        cls="tm-score",
    )
    return f.svg(
        "HausdorffDistance: the largest distance from a change point to the "
        f"nearest change point of the other set, {res['score']:g} points"
    )


def _segments(cps):
    b = [0, *cps, N]
    return list(zip(b[:-1], b[1:], strict=True))


def _iou(s, t):
    inter = max(0, min(s[1], t[1]) - max(s[0], t[0]))
    return inter / (max(s[1], t[1]) - min(s[0], t[0]))


def fig_covering() -> str:
    from tsseg.metrics import BidirectionalCovering, Covering

    cov = Covering().compute(TRUE_CPS, PRED_CPS, n_timepoints=N)["score"]
    bi = BidirectionalCovering().compute(TRUE_CPS, PRED_CPS, n_timepoints=N)
    true_s, pred_s = _segments(TRUE_CPS), _segments(PRED_CPS)
    f = Fig(196)
    yt, yp, h = 22, 104, 26
    f.row_label(yt + h / 2 + 4, "Truth")
    f.row_label(yp + h / 2 + 4, "Prediction")
    for i, (a, b) in enumerate(true_s):
        f.rect(
            f.x(a) + 1,
            yt,
            f.x(b) - f.x(a) - 2,
            h,
            f"tm-seg{i % 2}",
            rx=4,
            title=f"true segment [{a}, {b})",
        )
    for i, (a, b) in enumerate(pred_s):
        f.rect(
            f.x(a) + 1,
            yp,
            f.x(b) - f.x(a) - 2,
            h,
            f"tm-seg{i % 2}",
            rx=4,
            title=f"predicted segment [{a}, {b})",
        )
    total = 0.0
    terms = []
    for s in true_s:
        best = max(pred_s, key=lambda t: _iou(s, t))
        iou = _iou(s, best)
        total += (s[1] - s[0]) * iou
        terms.append(f"{s[1] - s[0]}·{_fmt(iou)}")
        xs, xp = f.x((s[0] + s[1]) / 2), f.x((best[0] + best[1]) / 2)
        f.line(
            xs,
            yt + h,
            xp,
            yp,
            "tm-link",
            title=f"[{s[0]}, {s[1]}) best matches [{best[0]}, {best[1]}): "
            f"IoU {iou:.3f}",
        )
        f.text(
            (xs + xp) / 2 + 6, (yt + h + yp) / 2 + 4, f"IoU {_fmt(iou)}", cls="tm-ink"
        )
    _check("Covering", total / N, cov)
    _check("Covering (ground truth)", total / N, bi["ground_truth_covering"])
    f.axis_ends(yp + h + 16)
    f.text(
        LABEL_X,
        168,
        f"Covering = ({' + '.join(terms)}) / {N:,} = {_fmt(cov)}",
        cls="tm-score",
    )
    f.text(
        LABEL_X,
        188,
        f"prediction covered by the truth: {_fmt(bi['prediction_covering'])} · "
        f"BidirectionalCovering (harmonic mean) = {_fmt(bi['score'])}",
        cls="tm-score",
    )
    return f.svg(
        "Covering: each true segment takes its best intersection over union with "
        f"a predicted segment, weighted by its length; Covering {_fmt(cov)}"
    )


# -------------------------------------------------------------------- states
SD_X1 = 532  # the right column holds the scores


def _score_table(f: Fig, rows: list[tuple[float, list[float]]], heads: list[str]):
    cols = [700 - 42 * (len(heads) - 1 - i) for i in range(len(heads))]
    for x, name in zip(cols, heads, strict=True):
        f.text(x, rows[0][0] - 22, name, cls="tm-head", anchor="middle")
    for y, values in rows:
        for x, v in zip(cols, values, strict=True):
            f.text(x, y + 4, _fmt(v), cls="tm-ink", anchor="middle")


def fig_permutation() -> str:
    from tsseg.metrics import (
        AdjustedMutualInformation,
        AdjustedRandIndex,
        NormalizedMutualInformation,
    )

    renamed = {0: 2, 1: 0, 2: 1}  # A -> 2, B -> 0, C -> 1
    pred_renamed = [(a, b, renamed[s]) for a, b, s in STATE_TRUE]
    pred_merged = [(a, b, 1 if s == 2 else s) for a, b, s in STATE_TRUE]
    y_true = _labels(STATE_TRUE)
    metrics = [
        AdjustedRandIndex(),
        AdjustedMutualInformation(),
        NormalizedMutualInformation(),
    ]
    f = Fig(150, x1=SD_X1)
    h = 24
    rows = [
        (46, "Truth", STATE_TRUE, STATE_NAMES),
        (84, "Renamed", pred_renamed, None),
        (122, "Merged", pred_merged, STATE_NAMES),
    ]
    table = []
    for y, label, segs, names in rows:
        f.row_label(y + 4, label)
        f.states(y - h / 2, h, segs, names, title=f"{label}: ")
        pred = _labels(segs)
        table.append((y, [m.compute(y_true, pred)["score"] for m in metrics]))
    _score_table(f, table, ["ARI", "AMI", "NMI"])
    return f.svg(
        "ARI, AMI and NMI compare labelings up to a renaming of the states: the "
        "renamed prediction scores 1, the prediction that merges states A and C "
        f"scores ARI {_fmt(table[2][1][0])}"
    )


def fig_weighted() -> str:
    from tsseg.metrics import (
        AdjustedRandIndex,
        NormalizedMutualInformation,
        WeightedAdjustedRandIndex,
        WeightedNormalizedMutualInformation,
    )

    alpha = 0.1  # the default of WARI and WNMI
    y_true = _labels(STATE_TRUE)
    late = y_true.copy()
    late[600:640] = 1  # the B -> A boundary placed 40 points late
    inner = y_true.copy()
    inner[430:470] = 0  # 40 points of A inside B
    errors = [(600, 640), (430, 470)]
    f = Fig(232, x1=SD_X1)
    h = 24
    f.row_label(34, "Truth")
    f.states(22, h, STATE_TRUE, STATE_NAMES, title="Truth: ")
    # weight 1 + alpha * distance to the nearest true boundary (ends included)
    bounds = np.array([0] + [b for _, b, _ in STATE_TRUE])
    t = np.arange(N)
    d = np.min(np.abs(t[:, None] - bounds[None, :]), axis=1)
    w = 1 + alpha * d
    top, base = 60, 124
    f.row_label(96, "Weight")
    pts = " ".join(
        f"{f.x(i)},{round(base - (base - top) * (w[i] - 1) / (w.max() - 1), 2)}"
        for i in range(0, N, 5)
    )
    f.add(
        f'<polygon class="tm-area" points="{f.x(0)},{base} {pts} {f.x(N - 1)},{base}"/>'
    )
    for a, b in errors:
        f.rect(
            f.x(a),
            top - 4,
            f.x(b) - f.x(a),
            base - top + 4,
            "tm-errband",
            rx=2,
            title=f"weights of [{a}, {b}): {w[a:b].min():g} to {w[a:b].max():g}",
        )
    f.add(f'<polyline class="tm-curve" points="{pts}"/>')
    f.line(f.x(0), base, f.x(N), base, "tm-base")
    f.text(f.x(0) - 4, base + 4, "1", anchor="end")
    f.text(f.x(0) - 4, top + 4, f"{w.max():g}", anchor="end")
    f.text(SD_X1 + 14, top + 12, f"w = 1 + {alpha:g}·d", cls="tm-ink")
    f.text(SD_X1 + 14, top + 30, "d: distance to the")
    f.text(SD_X1 + 14, top + 46, "nearest true boundary")
    preds = [(162, "Late", late, errors[0]), (202, "Inside", inner, errors[1])]
    table = []
    for y, label, pred, (a, b) in preds:
        f.row_label(y + 4, label)
        segs, start = [], 0
        for i in range(1, N + 1):
            if i == N or pred[i] != pred[start]:
                segs.append((start, i, int(pred[start])))
                start = i
        f.states(y - h / 2, h, segs, STATE_NAMES, title=f"{label}: ")
        f.rect(
            f.x(a) - 1,
            y - h / 2 - 3,
            f.x(b) - f.x(a) + 2,
            h + 6,
            "tm-errbox",
            rx=4,
            title=f"{label}: {b - a} wrong points in [{a}, {b})",
        )
        table.append(
            (
                y,
                [
                    m.compute(y_true, pred)["score"]
                    for m in (
                        AdjustedRandIndex(),
                        WeightedAdjustedRandIndex(),
                        NormalizedMutualInformation(),
                        WeightedNormalizedMutualInformation(),
                    )
                ],
            )
        )
    _score_table(f, table, ["ARI", "WARI", "NMI", "WNMI"])
    return f.svg(
        "WARI and WNMI weight each time point by 1 + 0.1 times its distance to the "
        "nearest true boundary: 40 wrong points next to a boundary cost less than "
        "40 wrong points inside a segment"
    )


SMS_TRUE = [
    (0, 200, 0),
    (200, 450, 1),
    (450, 600, 0),
    (600, 640, 2),
    (640, 800, 0),
    (800, 1000, 2),
]
SMS_PRED = [
    (0, 230, 0),
    (230, 320, 1),
    (320, 345, 2),
    (345, 435, 1),
    (435, 470, 2),
    (470, 590, 0),
    (590, 660, 1),
    (660, 800, 0),
    (800, 1000, 2),
]
SMS_EXPECTED = ["delay", "isolation", "transition", "missing"]


def fig_sms() -> str:
    from tsseg.metrics import StateMatchingScore

    metric = StateMatchingScore()
    res = metric.compute(_labels(SMS_TRUE), _labels(SMS_PRED), return_errors=True)
    errors = res["errors"]
    if [e["type"] for e in errors] != SMS_EXPECTED:
        raise ValueError(f"metric figure: SMS error types {errors}")
    f = Fig(196)
    h = 24
    f.row_label(34, "Truth")
    f.states(22, h, SMS_TRUE, STATE_NAMES, title="Truth: ")
    f.row_label(74, "Prediction")
    f.states(62, h, SMS_PRED, STATE_NAMES, title="Prediction: ")
    for k, e in enumerate(errors):
        a, b = e["start"], e["end"] + 1
        f.rect(
            f.x(a) - 1,
            59,
            f.x(b) - f.x(a) + 2,
            h + 6,
            "tm-errbox",
            rx=4,
            title=f"{e['type']}: [{a}, {b}), {e['size']} points, "
            f"penalty {e['penalty']:.2f}",
        )
        xc = (f.x(a) + f.x(b)) / 2
        y = 112 if k % 2 == 0 else 134
        f.line(xc, 89, xc, y - 12, "tm-guide")
        f.text(xc, y, e["type"], cls="tm-ink-strong", anchor="middle")
    size = sum(e["size"] for e in errors)
    penalty = sum(e["penalty"] for e in errors)
    _check("SMS", 1 - (size + penalty) / N, res["score"])
    f.text(
        LABEL_X,
        166,
        f"SMS = 1 − (L + P) / n = 1 − ({size} + {_fmt(penalty, 1)}) / {N:,} = "
        f"{_fmt(res['score'])}",
        cls="tm-score",
    )
    f.text(
        LABEL_X, 186, f"L = {size} wrong points, P = their penalties (default weights)"
    )
    return f.svg(
        "State Matching Score: the wrong points form error segments typed as "
        f"delay, isolation, transition and missing; SMS {_fmt(res['score'])}"
    )


FIGURES = {
    "f1": fig_f1,
    "gaussian_f1": fig_gaussian_f1,
    "hausdorff": fig_hausdorff,
    "covering": fig_covering,
    "permutation": fig_permutation,
    "weighted": fig_weighted,
    "sms": fig_sms,
}


class MetricFigureDirective(SphinxDirective):
    """``.. tsseg-metric-figure:: <name>``, one of :data:`FIGURES`."""

    required_arguments = 1
    has_content = False

    def run(self):
        name = self.arguments[0]
        if name not in FIGURES:
            raise self.error(f"unknown metric figure {name!r}: {sorted(FIGURES)}")
        # rebuild the page when the figures or the metrics change
        import tsseg.metrics

        self.env.note_dependency(__file__)
        for path in sorted(Path(tsseg.metrics.__file__).parent.glob("*.py")):
            self.env.note_dependency(str(path))
        svg = FIGURES[name]()
        return [nodes.raw("", f'<figure class="tm-fig">{svg}</figure>', format="html")]


def setup(app: Sphinx):
    app.add_directive("tsseg-metric-figure", MetricFigureDirective)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
