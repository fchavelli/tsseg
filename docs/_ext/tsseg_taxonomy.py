"""Interactive taxonomy of the tsseg detectors.

Sphinx extension providing the ``tsseg-taxonomy`` directive. It places every
detector of :mod:`tsseg.algorithms` on a grid whose columns are search
procedures (how a segmentation is searched for) and whose rows are homogeneity
criteria (what makes a segment homogeneous), the taxonomy of the TSB-SEG
benchmark.

The placement, complexity and reference of each method are declared below. The
capabilities shown by the page (unsupervised or guided, multivariate, optional
extra) are read from the estimator tags at build time, so they follow the code.
The build warns, hence fails under ``-W``, when a public detector is missing
from the taxonomy or when a detector page does not exist.

The directive emits a static list of links (shown without JavaScript) and the
data of the interactive page as JSON; ``_static/js/taxonomy.js`` renders it.
"""

from __future__ import annotations

import html
import json
import re

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------
SEARCHES = [
    {
        "key": "profile",
        "task": "cpd",
        "name": "Profile + peaks",
        "short": "A score at every time point, computed once; change points at its peaks.",
        "long": (
            "The series is scanned once to compute a score at every time point: "
            "the discrepancy between two adjacent windows, a novelty score, or how "
            "many nearest-neighbour arcs cross the point. Change points are placed "
            "at the peaks of this profile, above a threshold or the highest ones."
        ),
    },
    {
        "key": "bottomup",
        "task": "cpd",
        "name": "Bottom-up",
        "short": "From a fine partition, merge the most similar adjacent segments.",
        "long": (
            "Start from many small segments and repeatedly merge the two adjacent "
            "segments whose union costs least, until the next merge costs more "
            "than a penalty or the requested number of segments is reached."
        ),
    },
    {
        "key": "binseg",
        "task": "cpd",
        "name": "Binary segmentation",
        "short": "Top-down: split where the gain is largest, recursively.",
        "long": (
            "Find the single best split of the series, then search again inside "
            "each part. Wild and seeded binary segmentation search random or "
            "seeded sub-intervals instead, to recover change points close to "
            "each other."
        ),
    },
    {
        "key": "convex",
        "task": "cpd",
        "name": "Convex relaxation",
        "short": "ℓ₁ penalty on the jumps of a piecewise fit.",
        "long": (
            "Fit a piecewise model (level, trend) with an ℓ₁ penalty on its "
            "jumps, as in the fused lasso: most jumps are exactly zero, and the "
            "others are the change points."
        ),
    },
    {
        "key": "optimal",
        "task": "cpd",
        "dp": True,
        "name": "Optimal (DP)",
        "short": "Exact best segmentation (min over all segmentations).",
        "long": (
            "Minimise the sum of the segment costs over all segmentations, with a "
            "penalty per change point or a fixed number of them, by dynamic "
            "programming over the position of the last change point. PELT prunes "
            "the candidates that can no longer win."
        ),
    },
    {
        "key": "bayes",
        "task": "cpd",
        "dp": True,
        "name": "Bayesian (DP)",
        "short": "Posterior over segmentations (sum), exact by DP or sampled by MCMC.",
        "long": (
            "Put a prior on segmentations (product partition, or a left-to-right "
            "hidden Markov model) and compute the posterior probability of a "
            "change at each point: by the recursion of the optimal search with "
            "sums in place of minima, or by Markov chain Monte Carlo."
        ),
    },
    {
        "key": "states",
        "task": "sd",
        "dp": True,
        "name": "State sequence (DP)",
        "short": "A state per step or window, tied to the next one by a transition cost.",
        "long": (
            "One hidden state per time step (per window in TICC, per block of the "
            "reduced series in HDP-HSMM), linked to the next by a transition model "
            "or a switching cost. Viterbi, forward–backward or Gibbs sampling "
            "runs over the whole sequence, so a state can come back."
        ),
    },
    {
        "key": "seggroup",
        "task": "sd",
        "name": "Segment, then group",
        "short": "Change points first, then similar segments share a state.",
        "long": (
            "Detect change points first, then compare the segments: the segments "
            "that are alike receive the same state label."
        ),
    },
    {
        "key": "cluster",
        "task": "sd",
        "name": "Window clustering",
        "short": "Each window or point clustered on its own; time order ignored or in features only.",
        "long": (
            "Each time point or window is labelled by a clustering (k-means, "
            "Gaussian or Dirichlet process mixture). Time order is ignored, or "
            "enters only the representation, for instance through contrastive "
            "pairs of adjacent windows."
        ),
    },
]

CRITERIA = [
    {
        "key": "model",
        "tag": "model",
        "name": "Model of the observations",
        "short": "Likelihood or fit of a model (mean, variance, AR, trend, …).",
        "long": (
            "Two parts differ when a specified model fits them with different "
            "parameters: mean, median, covariance, precision graph, regression or "
            "autoregression, trend, subspace, local intrinsic dimension."
        ),
    },
    {
        "key": "distrib",
        "tag": "distribution",
        "name": "Distribution, model-free",
        "short": "Kernel, energy, Wasserstein, density ratio, classifier or graph.",
        "long": (
            "Empirical distributions are compared without a parametric model: "
            "characteristic kernel, energy distance, isolation kernel, density "
            "ratio, a classifier trained to tell the parts apart, or a graph."
        ),
    },
    {
        "key": "shapes",
        "tag": "subsequence shapes",
        "name": "Subsequence shapes",
        "short": "Nearest neighbours, classifiers or symbolic patterns of subsequences.",
        "long": (
            "z-normalised subsequences are compared to their nearest neighbours "
            "(matrix-profile arcs, k-NN classifiers) or described by symbolic "
            "patterns: two parts differ when their local shapes do."
        ),
    },
    {
        "key": "learned",
        "tag": "learned representation",
        "name": "Learned representations",
        "short": "Windows embedded by a trained network.",
        "long": (
            "A network (autoencoder, contrastive encoder, learned kernel) maps "
            "windows to embeddings, and the embeddings are compared."
        ),
    },
]

DP_NOTE = (
    "Same dynamic programme over segments: its <b>min</b> gives the best "
    "segmentation, its <b>sum</b> the posterior; with recurring states, Viterbi "
    "(min) and forward–backward (sum)."
)

# Costs of the ruptures-based detectors, by row: a detector whose cost is a user
# choice (``model=``) is drawn as one box spanning the rows of its costs. Cosine
# is l2 on unit-normalised vectors and linear is a regression model; only rbf (a
# characteristic kernel) compares distributions without a model.
COSTS = [
    ("model", "ℓ₁ ℓ₂ normal linear cosine"),
    ("distrib", "rbf"),
]

# The 26 detectors evaluated by TSB-SEG (NeurIPS 2026)
TSB_SEG = {
    "AMOC", "BinSeg", "BottomUp", "ChangeFinder", "PELT", "DynP", "Window",
    "Prophet", "GGS", "iCID", "IGTS", "FLUSS", "ESPRESSO", "TIRE", "ClaSP",
    "AutoPlait", "HDP-HSMM", "TICC", "CLaP", "E2USD", "Time2State", "BOCD",
    "E-Agglo", "TS-CP2", "tGLAD", "Hidalgo",
}  # fmt: skip

# Time complexity of one run as implemented, default settings, l2 cost; epochs
# and sampler / EM iterations omitted (as in the TSB-SEG figure).
TYPICAL = {"PELT", "BinSeg", "BottomUp"}


def _m(
    name, cls, page, search, row, approach, ref, cx=None, deep=False, status="tsseg"
):
    return {
        "name": name,
        "cls": cls,
        "page": page,
        "search": search,
        "row": row,
        "approach": approach,
        "ref": ref,
        "cx": cx,
        "deep": deep,
        "status": status,
    }


BAR = "bar"  # row of the detectors whose cost is chosen by the user

METHODS = [
    # ---- Change point detection: profile + peaks
    _m("ChangeFinder", "ChangeFinderDetector", "changefinder", "profile", "model",
       "two-stage autoregressive outlier scores", "Takeuchi & Yamanishi (2006), TKDE", "dn"),
    _m("SST", "SSTDetector", "sst", "profile", "model",
       "subspaces of past and future Hankel matrices",
       "Idé & Tsuda (2007), SDM; Weber & Lenz (2025), IEEE Access", "nl log l"),
    _m("tGLAD", "TGLADDetector", "tglad", "profile", "model",
       "changes of conditional independence graphs", "Imani & Shrivastava (2023)",
       "d^3 n", deep=True),
    _m("Window", "WindowDetector", "window", "profile", BAR,
       "sliding-window discrepancy", "Basseville & Nikiforov (1993)", "dnl"),
    _m("iCID", "ICIDDetector", "icid", "profile", "distrib",
       "isolation distributional kernel", "Cao et al. (2024), JAIR", "dn + n^2"),
    _m("RuLSIF", "RuLSIFDetector", "rulsif", "profile", "distrib",
       "relative density-ratio estimation", "Liu et al. (2013), Neural Netw.", "nb^3"),
    _m("FLUSS", "FLUSSDetector", "fluss", "profile", "shapes",
       "arc curve of the matrix profile", "Gharghabi et al. (2017), ICDM", "dn^2"),
    _m("ESPRESSO", "EspressoDetector", "espresso", "profile", "shapes",
       "entropy and shape of the matrix profile", "Deldari et al. (2020), UbiComp",
       "dn^2 log n"),
    _m("TIRE", "TireDetector", "tire", "profile", "learned",
       "autoencoder, time-invariant features", "De Ryck et al. (2021), IEEE TSP",
       "dnl", deep=True),
    _m("TS-CP2", "TSCP2Detector", "tscp2", "profile", "learned",
       "contrastive predictive coding", "Deldari et al. (2021), WWW", "dnl", deep=True),
    # ---- bottom-up
    _m("BottomUp", "BottomUpDetector", "bottomup", "bottomup", BAR,
       "greedy bottom-up merging", "Keogh et al. (2001), ICDM", "dn log n"),
    _m("E-Agglo", "EAggloDetector", "eagglo", "bottomup", "distrib",
       "agglomerative, energy distance", "Matteson & James (2014), JASA", "dn^2"),
    # ---- binary segmentation
    _m("GGS", "GreedyGaussianDetector", "ggs", "binseg", "model",
       "greedy Gaussian likelihood", "Hallac et al. (2019), ADAC", "n_CP d^3 n"),
    _m("IGTS", "InformationGainDetector", "igts", "binseg", "model",
       "information gain, top-down", "Sadri et al. (2017), PMC", "n_CP^2 dn"),
    _m("BinSeg", "BinSegDetector", "binseg", "binseg", BAR,
       "greedy top-down splitting", "Bai (1997); Fryzlewicz (2014)", "dn log n"),
    _m("Seeded BS", "SeededBinSegDetector", "seeded_binseg", "binseg", BAR,
       "best split of seeded or wild intervals",
       "Kovács et al. (2023), Biometrika; Fryzlewicz (2014), Ann. Stat.",
       "dn log n"),
    _m("changeforest", "ChangeForestDetector", "changeforest", "binseg", "distrib",
       "random forest classifier between segments", "Londschien et al. (2023), JMLR"),
    _m("ClaSP", "ClaspDetector", "clasp", "binseg", "shapes",
       "classification score profile", "Ermshaus et al. (2023), DMKD", "n_CP dn^2"),
    # ---- convex relaxation
    _m("Prophet", "ProphetDetector", "prophet", "convex", "model",
       "trend change points of Prophet", "Taylor & Letham (2018)", "dn"),
    _m("Cachalot", "CachalotDetector", "cachalot", "convex", "model",
       "Lasso candidates, then reduced dynamic programming",
       "Harchaoui & Lévy-Leduc (2007; 2010)", "d(n K + K^3)"),
    # ---- optimal (DP)
    _m("AMOC", "AmocDetector", "amoc", "optimal", "model",
       "single best split", "Killick et al. (2012)", "dn"),
    _m("PELT", "PeltDetector", "pelt", "optimal", BAR,
       "exact penalised search with pruning", "Killick et al. (2012), JASA", "dn"),
    _m("DynP", "DynpDetector", "dynp", "optimal", BAR,
       "exact dynamic programming", "Auger & Lawrence (1989)", "n_CP dn^2"),
    _m("KCPD", "KCPDDetector", "kcpd", "optimal", "distrib",
       "kernel change point detection; use PELT or DynP with model=\"rbf\"",
       "Arlot, Celisse & Harchaoui (2019), JMLR", status="deprecated"),
    # ---- Bayesian (DP)
    _m("BOCD", "BOCDDetector", "bocd", "bayes", "model",
       "Bayesian run-length posterior (offline run of an online method)",
       "Adams & MacKay (2007)", "dn + n^3"),
    _m("BEAST", "BeastDetector", "beast", "bayes", "model",
       "Bayesian model averaging of trend and seasonality", "Zhao et al. (2019), RSE",
       "dn"),
    # ---- State detection: state sequence (DP)
    _m("Gaussian HMM", "GaussianHMMDetector", "gaussian_hmm", "states", "model",
       "hidden Markov model fitted by EM", "Rabiner (1989)", "n_S n(n_S + d)"),
    _m("HDP-HSMM", "HdpHsmmDetector", "hdp_hsmm", "states", "model",
       "Bayesian nonparametric semi-Markov model", "Johnson & Willsky (2013), JMLR",
       "dn + n_S(n_S + d^2)"),
    _m("AutoPlait", "AutoPlaitDetector", "autoplait", "states", "model",
       "MDL-based regime mining", "Matsubara et al. (2014), SIGMOD", "dn"),
    _m("TICC", "TiccDetector", "ticc", "states", "model",
       "Toeplitz inverse covariance clustering", "Hallac et al. (2017), KDD",
       "n_S(n(dl)^2 + (dl)^3)"),
    _m("SNLDS", "SNLDSDetector", "snlds", "states", "learned",
       "switching nonlinear dynamical system (experimental)", "Dong et al. (2020), ICML",
       deep=True),
    _m("HMM", "HMMDetector", "hmm", "states", "model",
       "univariate hidden Markov model; use Gaussian HMM", "Rabiner (1989)",
       status="deprecated"),
    # ---- segment, then group
    _m("CLaP", "ClapDetector", "clap", "seggroup", "shapes",
       "classification label profile", "Ermshaus et al. (2024)",
       "n_CP n(nd + n_CP log n_CP)"),
    # ---- window clustering
    _m("Hidalgo", "HidalgoDetector", "hidalgo", "cluster", "model",
       "local intrinsic dimension", "Allegra et al. (2020), Sci. Rep.", "dn^2"),
    _m("Time2State", "Time2StateDetector", "time2state", "cluster", "learned",
       "contrastive encoder, clustering", "Wang et al. (2023), SIGMOD", "dnl", deep=True),
    _m("E2USD", "E2USDDetector", "e2usd", "cluster", "learned",
       "dual-view embeddings, clustering", "Lai et al. (2024), WWW", "dnl log l",
       deep=True),
    _m("Laplacian", "LaplacianDetector", "laplacian", "cluster", "distrib",
       "spectral clustering of a recurrence graph of windows",
       "McFee & Ellis (2014), ISMIR", "dn^2 + n^3"),
    _m("PaTSS", "PatssDetector", "patss", "cluster", "shapes",
       "sequential patterns of SAX words, then clustering",
       "Carpentier et al. (2024)", status="deprecated"),
]  # fmt: skip

# Methods of the TSB-SEG taxonomy that tsseg does not implement (no page): shown
# on demand, with their reference only.
LITERATURE = [
    ("MOSUM", "profile", "model", "Eichinger & Kirch (2018)"),
    ("subspace ID", "profile", "model", "Kawahara et al. (2007)"),
    ("KL-CPD", "profile", "learned", "Chang et al. (2019)"),
    ("TGUH", "bottomup", "model", "Fryzlewicz (2018)"),
    ("MDL greedy", "bottomup", "model", "Wu et al. (2024)"),
    ("NOT", "binseg", "model", "Baranowski et al. (2019)"),
    ("Inspect", "binseg", "model", "Wang & Samworth (2018)"),
    ("E-Divisive", "binseg", "distrib", "Matteson & James (2014)"),
    ("gSeg", "binseg", "distrib", "Chen & Zhang (2015)"),
    ("fast group LARS", "convex", "model", "Vert & Bleakley (2010)"),
    ("ℓ₁ trend filtering", "convex", "model", "Kim et al. (2009)"),
    ("FPOP / SNIP", "optimal", "model", "Maidstone et al. (2017)"),
    ("SMUCE", "optimal", "model", "Frick et al. (2014)"),
    ("BOCPDMS", "bayes", "model", "Knoblauch & Damoulas (2018)"),
    ("PPM (bcp)", "bayes", "model", "Barry & Hartigan (1993)"),
    ("change-point HMM", "bayes", "model", "Chib (1998)"),
    ("DPHMM", "bayes", "model", "Ko et al. (2015)"),
    ("GP-BOCPD", "bayes", "model", "Saatçi et al. (2010)"),
    ("rSLDS", "states", "model", "Linderman et al. (2017)"),
    ("sticky HDP-HMM", "states", "model", "Fox et al. (2011)"),
    ("BP-AR-HMM", "states", "model", "Fox et al. (2014)"),
    ("ClaSP2Feat", "seggroup", "model", "Ermshaus et al. (2025)"),
    ("optimal transport", "seggroup", "distrib", "Cheng et al. (2020)"),
    ("WK-means", "cluster", "distrib", "Horvath et al. (2024)"),
    ("ordinal patterns", "cluster", "shapes", "Sinn et al. (2013)"),
]

# Public detectors left out of the grid on purpose
EXCLUDED = {
    "RandomDetector": ("random", "random baseline for both tasks"),
}

TASKS = {"change_point_detection": "cpd", "state_detection": "sd"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
_CX_TOKEN = re.compile(r"n_CP|n_S|log|\^\d|[a-zA-Z]|\d|[()+]|\s+")


def complexity_html(expr: str) -> str:
    """``"n_CP dn^2"`` -> HTML of O(n_CP d n^2), variables in italics."""
    out = []
    for tok in _CX_TOKEN.findall(expr):
        if tok == "n_CP":
            out.append("<i>n</i><sub>CP</sub>")
        elif tok == "n_S":
            out.append("<i>n</i><sub>S</sub>")
        elif tok == "log":
            out.append("log\u2009")
        elif tok.startswith("^"):
            out.append(f"<sup>{tok[1:]}</sup>")
        elif tok == "l":
            out.append("<i>ℓ</i>")
        elif tok.isalpha():
            out.append(f"<i>{tok}</i>")
        elif tok.isspace():
            out.append("\u2009")
        elif tok == "+":
            out.append("\u2009+\u2009")
        else:
            out.append(tok)
    return "O(" + "".join(out).replace("\u2009\u2009", "\u2009") + ")"


def _extras_of(deps, extras_table) -> list[str]:
    """Smallest set of extras installing ``deps`` (unknown modules kept as is)."""
    if not deps:
        return []
    deps = set(deps)
    single = [e for e, mods in extras_table.items() if deps <= set(mods)]
    if single:
        return [min(single, key=lambda e: len(extras_table[e]))]
    found = []
    for dep in sorted(deps):
        owners = [e for e, mods in extras_table.items() if dep in mods]
        name = min(owners, key=lambda e: len(extras_table[e])) if owners else dep
        if name not in found:
            found.append(name)
    return found


def _tags(cls) -> dict:
    get = cls.get_class_tag
    return {
        "task": TASKS.get(get("detector_type")),
        "unsup": bool(get("capability:unsupervised", False)),
        "guided": bool(get("capability:semi_supervised", False)),
        "uni": bool(get("capability:univariate", False)),
        "mv": bool(get("capability:multivariate", False)),
        "deps": list(get("python_dependencies", None) or []),
    }


def build_data(env, docname: str) -> dict:
    """Taxonomy as a JSON-able dict, links relative to ``docname``."""
    import tsseg.algorithms as algorithms
    from tsseg.algorithms._dependencies import EXTRAS

    builder = env.app.builder
    search_task = {s["key"]: s["task"] for s in SEARCHES}

    def url(page):
        target = f"detectors/{page}"
        if target not in env.found_docs:
            logger.warning(
                "tsseg-taxonomy: no page %s", target, location=(docname, None)
            )
        return builder.get_relative_uri(docname, target)

    methods = []
    seen = set()
    for spec in METHODS:
        cls = getattr(algorithms, spec["cls"], None)
        if cls is None or spec["cls"] not in algorithms.__all__:
            logger.warning(
                "tsseg-taxonomy: %s is not public in tsseg.algorithms",
                spec["cls"],
                location=(docname, None),
            )
            continue
        seen.add(spec["cls"])
        tags = _tags(cls)
        if tags["task"] != search_task[spec["search"]]:
            logger.warning(
                "tsseg-taxonomy: %s is placed under a %s search but its "
                "detector_type says %s",
                spec["cls"],
                search_task[spec["search"]],
                tags["task"],
                location=(docname, None),
            )
        guidance = None
        if tags["guided"]:
            guidance = "opt" if tags["unsup"] else "req"
        cx = spec["cx"]
        methods.append(
            {
                "id": spec["cls"],
                "name": spec["name"],
                "cls": spec["cls"],
                "url": url(spec["page"]),
                "task": tags["task"],
                "search": spec["search"],
                "rows": [r for r, _ in COSTS] if spec["row"] == BAR else [spec["row"]],
                "bar": spec["row"] == BAR,
                "cx": complexity_html(cx) if cx else None,
                "typical": spec["name"] in TYPICAL,
                "guidance": guidance,
                "unsup": tags["unsup"],
                "guided": tags["guided"],
                "uni": tags["uni"],
                "mv": tags["mv"],
                "extras": _extras_of(tags["deps"], EXTRAS),
                "deep": spec["deep"],
                "bench": spec["name"] in TSB_SEG,
                "status": spec["status"],
                "approach": spec["approach"],
                "ref": spec["ref"],
            }
        )

    for i, (name, search, row, ref) in enumerate(LITERATURE):
        methods.append(
            {
                "id": f"lit-{i}",
                "name": name,
                "task": search_task[search],
                "search": search,
                "rows": [row],
                "bar": False,
                "status": "literature",
                "ref": ref,
            }
        )

    missing = [n for n in algorithms.__all__ if n not in seen and n not in EXCLUDED]
    for name in missing:
        logger.warning(
            "tsseg-taxonomy: %s is in tsseg.algorithms.__all__ but not in the "
            "taxonomy (docs/_ext/tsseg_taxonomy.py)",
            name,
            location=(docname, None),
        )

    excluded = [
        {"cls": cls, "url": url(page), "why": why}
        for cls, (page, why) in EXCLUDED.items()
    ]
    return {
        "searches": SEARCHES,
        "criteria": CRITERIA,
        "costs": [{"row": r, "label": label} for r, label in COSTS],
        "methods": methods,
        "excluded": excluded,
        "dpNote": DP_NOTE,
        "tsbseg": "https://github.com/fchavelli/TSB-SEG",
    }


def fallback_html(data: dict) -> str:
    """Plain nested list of links, shown when JavaScript is off."""
    crit = {c["key"]: c["name"] for c in data["criteria"]}
    parts = ['<div class="tx-fallback">']
    for task, title in (("cpd", "Change point detection"), ("sd", "State detection")):
        parts.append(f"<h3>{title}</h3><dl>")
        for s in (s for s in data["searches"] if s["task"] == task):
            items = [
                m
                for m in data["methods"]
                if m["search"] == s["key"] and m["status"] != "literature"
            ]
            links = ", ".join(
                f'<a href="{html.escape(m["url"])}">{html.escape(m["name"])}</a>'
                f" ({html.escape(' / '.join(crit[r] for r in m['rows']))})"
                for m in items
            )
            parts.append(f"<dt>{html.escape(s['name'])}</dt><dd>{links}</dd>")
        parts.append("</dl>")
    parts.append("</div>")
    return "".join(parts)


class TaxonomyDirective(SphinxDirective):
    has_content = False

    def run(self):
        data = build_data(self.env, self.env.docname)
        payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
        markup = (
            '<div class="tx-app" id="tsseg-taxonomy" data-state="loading">'
            f"{fallback_html(data)}</div>"
            '<script type="application/json" id="tsseg-taxonomy-data">'
            f"{payload}</script>"
        )
        return [nodes.raw("", markup, format="html")]


class TaxonomyPreviewDirective(SphinxDirective):
    """Mini-map of the grid, one square per detector, linking to the page."""

    has_content = False
    required_arguments = 1  # docname of the taxonomy page

    def run(self):
        data = build_data(self.env, self.env.docname)
        target = self.env.app.builder.get_relative_uri(
            self.env.docname, self.arguments[0]
        )
        order = [s["key"] for s in SEARCHES]
        task = {s["key"]: s["task"] for s in SEARCHES}
        shown = [m for m in data["methods"] if m["status"] == "tsseg"]
        parts = [
            f'<a class="txp" href="{html.escape(target)}" '
            'aria-label="Open the interactive taxonomy"><div class="txp-grid">'
        ]
        parts += [f'<span class="txp-head is-{task[k]}"></span>' for k in order]
        for crit in CRITERIA:
            for key in order:
                dots = "".join(
                    f'<i class="is-{m["task"]}" title="{html.escape(m["name"])}"></i>'
                    for m in shown
                    if m["search"] == key and m["rows"][0] == crit["key"]
                )
                parts.append(f'<span class="txp-cell">{dots}</span>')
        parts.append(
            '</div><div class="txp-foot"><span>search procedure →</span>'
            "<span>↓ homogeneity criterion</span></div></a>"
        )
        return [nodes.raw("", "".join(parts), format="html")]


def _page_context(app: Sphinx, pagename, templatename, context, doctree):
    """Taxonomy assets on the taxonomy page; placement card on detector pages."""
    if pagename.startswith("detectors/"):
        page = pagename.split("/", 1)[1]
        search = {s["key"]: s for s in SEARCHES}
        crit = {c["key"]: c for c in CRITERIA}
        cards = []
        for spec in METHODS:
            if spec["page"] != page:
                continue
            rows = (
                " or ".join(crit[r]["tag"] for r, _ in COSTS) + ", by cost"
                if spec["row"] == BAR
                else crit[spec["row"]]["tag"]
            )
            cards.append(
                {
                    "name": spec["name"],
                    "task": search[spec["search"]]["task"],
                    "search": search[spec["search"]]["name"],
                    "criterion": rows[0].upper() + rows[1:],
                    "url": context["pathto"]("taxonomy") + "#m=" + spec["cls"],
                }
            )
        context["tsseg_taxonomy"] = cards
    if doctree is not None and any(
        "tsseg-taxonomy-data" in node.astext() for node in doctree.findall(nodes.raw)
    ):
        app.add_css_file("css/taxonomy.css")
        app.add_js_file("js/taxonomy.js", defer="defer")


def setup(app: Sphinx):
    app.add_directive("tsseg-taxonomy", TaxonomyDirective)
    app.add_directive("tsseg-taxonomy-preview", TaxonomyPreviewDirective)
    app.connect("html-page-context", _page_context)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
