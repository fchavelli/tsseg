/* Interactive taxonomy of the tsseg detectors.
 *
 * Renders the data emitted by the ``tsseg-taxonomy`` directive
 * (docs/_ext/tsseg_taxonomy.py): a grid of search procedures (columns) by
 * homogeneity criteria (rows), a sortable list, filters, search and a preview
 * card. Every detector links to its documentation page.
 */
(function () {
  "use strict";

  const app = document.getElementById("tsseg-taxonomy");
  const dataEl = document.getElementById("tsseg-taxonomy-data");
  if (!app || !dataEl) return;

  const D = JSON.parse(dataEl.textContent);
  const SEARCH = Object.fromEntries(D.searches.map((s) => [s.key, s]));
  const CRIT = Object.fromEntries(D.criteria.map((c) => [c.key, c]));
  const METHODS = D.methods;
  const BY_ID = Object.fromEntries(METHODS.map((m) => [m.id, m]));
  const TASK = { cpd: "Change point detection", sd: "State detection" };
  const TASK_SHORT = { cpd: "Change points", sd: "States" };

  // ------------------------------------------------------------------ utils
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v; // trusted: built by our extension
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c.nodeType ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  const svg = (inner, vb = "0 0 120 46") =>
    `<svg viewBox="${vb}" aria-hidden="true" focusable="false">${inner}</svg>`;
  const isTsseg = (m) => m.status !== "literature";
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const critLabel = (m) =>
    m.bar ? `${cap(m.rows.map((r) => CRIT[r].tag).join(" or "))}, by cost` : CRIT[m.rows[0]].name;

  // ---------------------------------------------------------- micro-figures
  const arc = (x1, x2, y, k = 0.5) =>
    `M${x1} ${y} Q${(x1 + x2) / 2} ${y - (x2 - x1) * k} ${x2} ${y}`;
  const NODES = [8, 25, 42, 59, 76, 93, 110];
  function arcs(weighted) {
    let out = "";
    NODES.forEach((a, i) =>
      NODES.slice(i + 1, i + 4).forEach((b, j) => {
        const w = weighted ? [0.9, 0.35, 0.15, 0.6, 0.25][(i + j) % 5] : 0;
        out += weighted
          ? `<path class="mf-accent" d="${arc(a, b, 40)}" style="stroke-width:${(0.5 + 2.2 * w).toFixed(2)};opacity:${(0.25 + 0.7 * w).toFixed(2)}"/>`
          : `<path class="mf-faint" d="${arc(a, b, 40)}"/>`;
      }),
    );
    return out;
  }
  const JITTER = [2, -3, 1, 3, -2, 0, -3, 2, 1, -1, 3, -2, 0, 2, -3, 1, -1, 2, 0];
  const MF = {
    profile: svg(
      `<line class="mf-faint" x1="4" y1="41" x2="116" y2="41"/>` +
        `<path class="mf-accent mf-bold" d="M4 37C14 37 20 36 26 31C32 25 36 9 42 9C48 9 51 28 58 33C64 37 70 35 76 30C82 24 86 13 90 13C95 13 98 31 104 35C108 37 112 37 116 37"/>` +
        `<line class="mf-cp" x1="42" y1="2" x2="42" y2="44"/><line class="mf-cp" x1="90" y1="2" x2="90" y2="44"/>` +
        `<circle class="mf-dot" cx="42" cy="9" r="2.8"/><circle class="mf-dot" cx="90" cy="13" r="2.8"/>`,
    ),
    bottomup: svg(
      rowBars(36, [16, 28, 40, 53, 66, 79, 92, 104, 116], "mf-fill-faint") +
        rowBars(22, [16, 40, 66, 92, 116], "mf-fill-muted") +
        rowBars(8, [16, 40, 92, 116], "mf-fill-accent") +
        `<path class="mf-ink" d="M7 42V7M3.5 11L7 6.5L10.5 11"/>`,
    ),
    binseg: svg(
      rowBars(8, [16, 116], "mf-fill-faint") +
        rowBars(22, [16, 66, 116], "mf-fill-muted") +
        rowBars(36, [16, 40, 66, 116], "mf-fill-accent") +
        `<path class="mf-ink" d="M7 4V39M3.5 35L7 39.5L10.5 35"/>`,
    ),
    convex: svg(
      JITTER.map((j, i) => {
        const x = 6 + i * 6;
        const lvl = x < 42 ? 32 : x < 80 ? 13 : 25;
        return `<circle class="mf-fill-muted" cx="${x}" cy="${lvl + j}" r="1.7"/>`;
      }).join("") + `<path class="mf-accent mf-bold" d="M4 32H42V13H80V25H116"/>`,
    ),
    optimal: svg(
      arcs(false) +
        `<path class="mf-accent mf-bold" d="${arc(8, 42, 40)} ${arc(42, 93, 40, 0.42)} ${arc(93, 110, 40)}"/>` +
        NODES.map((x) => `<circle class="mf-fill-ink" cx="${x}" cy="40" r="2"/>`).join(""),
    ),
    bayes: svg(
      arcs(true) +
        NODES.map(
          (x, i) =>
            `<circle class="mf-fill-accent" cx="${x}" cy="40" r="${[2, 3.2, 1.4, 2.8, 1.6, 3, 2][i]}"/>`,
        ).join(""),
    ),
    states: (() => {
      const xs = [20, 37, 54, 71, 88, 105];
      const path = ["A", "A", "B", "B", "B", "A"];
      const y = { A: 11, B: 35 };
      let s = `<text class="mf-label" x="3" y="14">A</text><text class="mf-label" x="3" y="38">B</text>`;
      for (let i = 0; i < xs.length - 1; i++)
        for (const a of ["A", "B"])
          for (const b of ["A", "B"])
            s += `<line class="mf-faint" x1="${xs[i]}" y1="${y[a]}" x2="${xs[i + 1]}" y2="${y[b]}"/>`;
      s += `<path class="mf-accent mf-bold" d="M${xs.map((x, i) => `${x} ${y[path[i]]}`).join("L")}"/>`;
      xs.forEach((x, i) =>
        ["A", "B"].forEach((st) => {
          const on = path[i] === st;
          s += `<circle class="${on ? "mf-fill-accent" : "mf-node"}" cx="${x}" cy="${y[st]}" r="${on ? 3.4 : 2.6}"/>`;
        }),
      );
      return svg(s);
    })(),
    seggroup: svg(
      [
        [6, 32],
        [35, 58],
        [61, 90],
        [93, 114],
      ]
        .map(
          ([a, b], i) =>
            `<rect class="mf-fill-muted" x="${a}" y="5" width="${b - a}" height="7" rx="2"/>` +
            `<path class="mf-ink-thin" d="M${(a + b) / 2} 15V26M${(a + b) / 2 - 3} 23L${(a + b) / 2} 27L${(a + b) / 2 + 3} 23"/>` +
            `<rect class="${i % 2 ? "mf-fill-alt" : "mf-fill-accent"}" x="${a}" y="31" width="${b - a}" height="7" rx="2"/>`,
        )
        .join(""),
    ),
    cluster: svg(
      `<path class="mf-ink-thin" d="M3 26C7 18 10 18 13 26S19 34 23 26S29 14 33 22S39 34 43 26S49 20 53 24"/>` +
        `<rect class="mf-win" x="3" y="13" width="17" height="24" rx="2"/><rect class="mf-win" x="20" y="13" width="17" height="24" rx="2"/><rect class="mf-win" x="37" y="13" width="17" height="24" rx="2"/>` +
        `<path class="mf-ink" d="M60 25H72M68.5 21.5L72.5 25L68.5 28.5"/>` +
        [
          [84, 14],
          [90, 10],
          [92, 17],
          [86, 21],
          [80, 9],
        ]
          .map(([x, y]) => `<circle class="mf-fill-accent" cx="${x}" cy="${y}" r="2.4"/>`)
          .join("") +
        [
          [104, 31],
          [110, 36],
          [100, 37],
          [108, 27],
          [114, 32],
        ]
          .map(([x, y]) => `<circle class="mf-fill-alt" cx="${x}" cy="${y}" r="2.4"/>`)
          .join(""),
    ),
  };
  function rowBars(y, cuts, cls) {
    let s = "";
    for (let i = 0; i < cuts.length - 1; i++)
      s += `<rect class="${cls}" x="${cuts[i] + 0.8}" y="${y - 3}" width="${cuts[i + 1] - cuts[i] - 1.6}" height="6.5" rx="1.6"/>`;
    return s;
  }
  const RF = {
    model: svg(
      [
        [10, 31],
        [19, 26],
        [28, 29],
        [37, 22],
        [47, 24],
        [57, 17],
        [66, 20],
        [76, 13],
        [86, 15],
        [96, 8],
        [106, 11],
      ]
        .map(([x, y]) => `<circle class="mf-fill-muted" cx="${x}" cy="${y}" r="2"/>`)
        .join("") + `<path class="mf-accent mf-bold" d="M5 32L112 7"/>`,
      "0 0 120 40",
    ),
    distrib: svg(
      [6, 13, 22, 30, 17, 9].map((v, i) => `<rect class="mf-fill-accent" x="${6 + i * 7.5}" y="${36 - v}" width="6" height="${v}" rx="1"/>`).join("") +
        `<text class="mf-neq" x="60" y="27" text-anchor="middle">≠</text>` +
        [24, 12, 7, 11, 22, 29].map((v, i) => `<rect class="mf-fill-muted" x="${70 + i * 7.5}" y="${36 - v}" width="6" height="${v}" rx="1"/>`).join(""),
      "0 0 120 40",
    ),
    shapes: svg(
      `<path class="mf-ink-thin" d="M4 30C10 30 12 14 18 14S26 30 32 30S44 28 50 27S60 30 66 30C72 30 74 14 80 14S88 30 94 30S108 26 116 28"/>` +
        `<path class="mf-accent mf-dash" d="M18 11Q49 -10 80 11"/>` +
        `<path class="mf-accent mf-dash" d="M32 33Q63 46 94 33"/>`,
      "0 0 120 40",
    ),
    learned: svg(
      `<path class="mf-ink-thin" d="M3 20C7 10 10 10 14 20S21 30 25 20S32 10 36 20"/>` +
        `<path class="mf-ink" d="M40 20H47M44 17L47.5 20L44 23"/>` +
        `<path class="mf-fill-muted" d="M51 6L75 13V27L51 34Z"/>` +
        `<path class="mf-ink" d="M79 20H86M83 17L86.5 20L83 23"/>` +
        [8, 17, 26].map((y) => `<rect class="mf-fill-accent" x="91" y="${y}" width="8" height="7" rx="1.5"/>`).join(""),
      "0 0 120 40",
    ),
  };

  const ICON = {
    search: `<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5"/><path d="M13 13l4.5 4.5"/></svg>`,
    grid: `<svg viewBox="0 0 20 20" aria-hidden="true"><rect x="3" y="3" width="5.5" height="5.5" rx="1.2"/><rect x="11.5" y="3" width="5.5" height="5.5" rx="1.2"/><rect x="3" y="11.5" width="5.5" height="5.5" rx="1.2"/><rect x="11.5" y="11.5" width="5.5" height="5.5" rx="1.2"/></svg>`,
    list: `<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 5h10M7 10h10M7 15h10"/><circle cx="3.6" cy="5" r="1.1"/><circle cx="3.6" cy="10" r="1.1"/><circle cx="3.6" cy="15" r="1.1"/></svg>`,
    pkg: `<svg class="tx-pkg" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.8l5.6 3v6.4L8 14.2l-5.6-3V4.8z"/><path d="M2.6 4.9L8 7.8l5.4-2.9M8 7.8v6.2"/></svg>`,
    arrow: `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8.5 4.5L12 8l-3.5 3.5"/></svg>`,
    close: `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 4l8 8M12 4l-8 8"/></svg>`,
  };

  // ------------------------------------------------------------------ state
  const FILTERS = [
    { key: "unsup", label: "Unsupervised", hint: "Runs without being told the number of change points or states", test: (m) => m.unsup },
    { key: "guided", label: "Guided", hint: "Accepts the number of change points or states in its constructor (n_cps, n_states, …)", test: (m) => m.guided },
    { key: "mv", label: "Multivariate", hint: "Accepts series with several channels", test: (m) => m.mv },
    { key: "core", label: "No extra install", hint: "Works with a plain pip install tsseg", test: (m) => !m.extras.length },
    { key: "deep", label: "Deep learning", hint: "Trains a neural network", test: (m) => m.deep },
    { key: "bench", label: "In TSB-SEG", hint: "Evaluated by the TSB-SEG benchmark (NeurIPS 2026)", test: (m) => m.bench },
  ];
  const state = {
    q: "",
    task: "all",
    filters: new Set(),
    lit: false,
    depr: false,
    view: null,
    focus: null, // "search:<key>" | "row:<key>"
    spot: null, // id of a method to point at once, from #m=<id>
    sort: { key: "name", dir: 1 },
  };

  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    state.q = p.get("q") || "";
    state.task = ["cpd", "sd"].includes(p.get("task")) ? p.get("task") : "all";
    state.filters = new Set(
      (p.get("f") || "").split(",").filter((k) => FILTERS.some((f) => f.key === k)),
    );
    state.lit = p.get("lit") === "1";
    state.depr = p.get("depr") === "1";
    state.view = ["grid", "list"].includes(p.get("view")) ? p.get("view") : null;
    state.spot = BY_ID[p.get("m")] ? p.get("m") : null;
    const fo = p.get("focus") || "";
    const [kind, key] = fo.split(":");
    state.focus =
      (kind === "search" && SEARCH[key]) || (kind === "row" && CRIT[key]) ? fo : null;
  }
  function writeHash() {
    const p = new URLSearchParams();
    if (state.q) p.set("q", state.q);
    if (state.task !== "all") p.set("task", state.task);
    if (state.filters.size) p.set("f", [...state.filters].join(","));
    if (state.lit) p.set("lit", "1");
    if (state.depr) p.set("depr", "1");
    if (state.view) p.set("view", state.view);
    if (state.focus) p.set("focus", state.focus);
    const s = p.toString();
    history.replaceState(null, "", s ? `#${s}` : location.pathname + location.search);
  }

  const norm = (s) =>
    String(s || "")
      .normalize("NFD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase();
  const HAY = Object.fromEntries(
    METHODS.map((m) => [
      m.id,
      norm(
        [
          m.name,
          m.cls,
          m.approach,
          m.ref,
          SEARCH[m.search].name,
          ...m.rows.map((r) => CRIT[r].name),
          m.bar ? D.costs.map((c) => c.label).join(" ") : "",
          TASK[m.task],
          m.task === "cpd" ? "cpd change point" : "sd state segmentation",
          ...(m.extras || []),
        ].join(" "),
      ),
    ]),
  );

  /** "hide" (not shown at all), "dim" (shown, does not match) or "on". */
  function status(m) {
    if (m.status === "literature" && !state.lit) return "hide";
    if (m.status === "deprecated" && !state.depr) return "hide";
    if (state.task !== "all" && m.task !== state.task) return "dim";
    if (state.filters.size) {
      if (!isTsseg(m)) return "dim";
      for (const f of FILTERS) if (state.filters.has(f.key) && !f.test(m)) return "dim";
    }
    if (state.q) {
      const tokens = norm(state.q).split(/\s+/).filter(Boolean);
      if (!tokens.every((t) => HAY[m.id].includes(t))) return "dim";
    }
    return "on";
  }

  // --------------------------------------------------------------- toolbar
  const input = h("input", {
    type: "search",
    class: "tx-search-input",
    placeholder: "Search methods, ideas, references…",
    "aria-label": "Search methods",
    autocomplete: "off",
    spellcheck: "false",
  });
  const searchBox = h(
    "label",
    { class: "tx-search" },
    h("span", { class: "tx-icon", html: ICON.search }),
    input,
    h("kbd", { class: "tx-kbd", title: "Press / to search" }, "/"),
  );

  function segmented(name, options, get, set) {
    const group = h("div", { class: "tx-seg", role: "radiogroup", "aria-label": name });
    const buttons = options.map(([value, label, extra]) =>
      h(
        "button",
        {
          type: "button",
          role: "radio",
          class: "tx-seg-btn",
          dataset: { value },
          onclick: () => {
            set(value);
            update();
          },
        },
        extra ? h("span", { class: "tx-icon", html: extra }) : null,
        label,
      ),
    );
    group.append(...buttons);
    group.sync = () =>
      buttons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.value === get())));
    return group;
  }
  const taskSeg = segmented(
    "Task",
    [
      ["all", "All tasks"],
      ["cpd", "Change points", '<span class="tx-dot tx-dot-cpd"></span>'],
      ["sd", "States", '<span class="tx-dot tx-dot-sd"></span>'],
    ],
    () => state.task,
    (v) => (state.task = v),
  );
  const viewSeg = segmented(
    "View",
    [
      ["grid", "Grid", ICON.grid],
      ["list", "List", ICON.list],
    ],
    () => currentView(),
    (v) => (state.view = v),
  );

  const filterButtons = FILTERS.map((f) =>
    h(
      "button",
      {
        type: "button",
        class: "tx-filter",
        title: f.hint,
        "aria-pressed": "false",
        dataset: { key: f.key },
        onclick: () => {
          state.filters.has(f.key) ? state.filters.delete(f.key) : state.filters.add(f.key);
          update();
        },
      },
      f.label,
    ),
  );
  function toggle(key, label, hint) {
    const box = h("input", {
      type: "checkbox",
      role: "switch",
      onchange: (e) => {
        state[key] = e.target.checked;
        update();
      },
    });
    const el = h("label", { class: "tx-switch", title: hint }, box, h("span", { class: "tx-switch-track" }), label);
    el.sync = () => (box.checked = state[key]);
    return el;
  }
  const litToggle = toggle("lit", "Literature", "Also show the methods of the taxonomy that tsseg does not implement");
  const deprToggle = toggle("depr", "Deprecated", "Also show the deprecated detectors");
  const count = h("span", { class: "tx-count", "aria-live": "polite" });
  const reset = h(
    "button",
    {
      type: "button",
      class: "tx-reset",
      onclick: () => {
        Object.assign(state, { q: "", task: "all", lit: false, depr: false, focus: null });
        state.filters.clear();
        input.value = "";
        update();
      },
    },
    "Reset",
  );

  const toolbar = h(
    "div",
    { class: "tx-toolbar" },
    h("div", { class: "tx-toolbar-row" }, searchBox, taskSeg, viewSeg),
    h(
      "div",
      { class: "tx-toolbar-row tx-toolbar-filters" },
      h("span", { class: "tx-label" }, "Filter"),
      filterButtons,
      h("span", { class: "tx-sep", "aria-hidden": "true" }),
      h("span", { class: "tx-label" }, "Show"),
      litToggle,
      deprToggle,
      h("span", { class: "tx-spacer" }),
      count,
      reset,
    ),
  );

  // ----------------------------------------------------------- focus panel
  const focusPanel = h("div", { class: "tx-focus", hidden: true, "aria-live": "polite" });
  function renderFocus() {
    focusPanel.replaceChildren();
    if (!state.focus) {
      focusPanel.hidden = true;
      return;
    }
    const [kind, key] = state.focus.split(":");
    const item = kind === "search" ? SEARCH[key] : CRIT[key];
    const inIt = METHODS.filter(
      (m) => isTsseg(m) && status(m) !== "hide" && (kind === "search" ? m.search === key : m.rows.includes(key)),
    );
    const kicker =
      kind === "search"
        ? `Search procedure · ${TASK[item.task]}`
        : "Homogeneity criterion";
    focusPanel.append(
      h(
        "div",
        { class: `tx-focus-fig ${kind === "search" ? `is-${item.task}` : "is-row"}`, html: kind === "search" ? MF[key] : RF[key] },
      ),
      h(
        "div",
        { class: "tx-focus-body" },
        h("div", { class: "tx-kicker" }, kicker),
        h("h3", { class: "tx-focus-title" }, item.name),
        h("p", {}, item.long),
        inIt.length
          ? h(
              "div",
              { class: "tx-focus-list" },
              h("span", { class: "tx-label" }, `In tsseg (${inIt.length})`),
              inIt.map((m) => h("a", { href: m.url, class: `tx-mini is-${m.task}` }, m.name)),
            )
          : h("div", { class: "tx-focus-list tx-muted" }, "No tsseg detector here yet."),
      ),
      h(
        "button",
        {
          type: "button",
          class: "tx-icon-btn tx-focus-close",
          "aria-label": "Clear the selection",
          html: ICON.close,
          onclick: () => {
            state.focus = null;
            update();
          },
        },
      ),
    );
    focusPanel.hidden = false;
  }
  function setFocus(value) {
    state.focus = state.focus === value ? null : value;
    update();
  }

  // ------------------------------------------------------------------ chips
  const chipEls = []; // [element, method]
  function badges(m) {
    const out = [];
    if (m.guidance)
      out.push(
        h(
          "span",
          {
            class: `tx-k tx-k-${m.guidance}`,
            title:
              m.guidance === "req"
                ? "Needs the number of change points or states"
                : "Accepts the number of change points or states (optional)",
          },
          "K",
        ),
      );
    if (m.extras && m.extras.length)
      out.push(h("span", { class: "tx-badge-pkg", title: `Needs an extra: tsseg[${m.extras.join(",")}]`, html: ICON.pkg }));
    return out;
  }
  function chip(m, inBar = false) {
    const lit = !isTsseg(m);
    const cls = [
      "tx-chip",
      `is-${m.task}`,
      lit ? "is-lit" : "",
      m.status === "deprecated" ? "is-depr" : "",
      inBar ? "in-bar" : "",
    ].join(" ");
    const marks = lit ? [] : badges(m);
    const body = [
      h("span", { class: "tx-chip-name" }, m.name),
      m.cx || marks.length || m.status === "deprecated"
        ? h(
            "span",
            { class: "tx-chip-meta" },
            m.cx
              ? h("span", { class: "tx-chip-cx", html: m.cx + (m.typical ? '<span class="tx-star">*</span>' : "") })
              : h("span", {}),
            m.status === "deprecated" ? h("span", { class: "tx-chip-tag" }, "deprecated") : null,
            marks.length ? h("span", { class: "tx-chip-marks" }, marks) : null,
          )
        : null,
    ];
    const el = lit
      ? h("span", { class: cls, tabindex: "0", dataset: { id: m.id } }, body)
      : h("a", { class: cls, href: m.url, dataset: { id: m.id } }, body);
    el.addEventListener("pointerenter", () => showCard(m, el));
    el.addEventListener("pointerleave", hideCard);
    el.addEventListener("focus", () => showCard(m, el));
    el.addEventListener("blur", hideCard);
    chipEls.push([el, m]);
    return el;
  }

  // ------------------------------------------------------------------- grid
  // Column layout: row header, 4 CPD searches, gap, 2 CPD DP searches, gap,
  // 1 SD DP search, gap, 2 SD searches. Spacer columns frame the DP group.
  const ORDER = ["profile", "bottomup", "binseg", "convex", "optimal", "bayes", "states", "seggroup", "cluster"];
  const COL = { profile: 2, bottomup: 3, binseg: 4, convex: 5, optimal: 7, bayes: 8, states: 10, seggroup: 12, cluster: 13 };
  const ROW = Object.fromEntries(D.criteria.map((c, i) => [c.key, 3 + i]));
  const LAST = 3 + D.criteria.length; // row of the DP note
  const colHeads = {};
  const rowHeads = {};
  const cells = []; // [element, searchKey, rowKey]
  const bars = []; // [elements, members]

  function buildGrid() {
    const grid = h("div", { class: "tx-grid", role: "group", "aria-label": "Taxonomy grid" });
    const place = (el, col, row, colSpan = 1, rowSpan = 1) => {
      el.style.gridColumn = `${col} / span ${colSpan}`;
      el.style.gridRow = `${row} / span ${rowSpan}`;
      grid.append(el);
      return el;
    };

    // DP group frame, behind the cells
    place(h("div", { class: "tx-halo", "aria-hidden": "true" }), COL.optimal, 2, COL.states - COL.optimal + 1, LAST - 1);

    // task bands
    const nTask = (t) => METHODS.filter((m) => m.task === t && m.status === "tsseg").length;
    place(
      h("div", { class: "tx-band is-cpd" }, h("span", {}, TASK.cpd), h("span", { class: "tx-band-n" }, `${nTask("cpd")} detectors`)),
      2,
      1,
      COL.bayes - 1,
    );
    place(
      h("div", { class: "tx-band is-sd" }, h("span", {}, TASK.sd), h("span", { class: "tx-band-n" }, `${nTask("sd")} detectors`)),
      COL.states,
      1,
      COL.cluster - COL.states + 1,
    );

    // corner
    place(
      h(
        "div",
        { class: "tx-corner", "aria-hidden": "true" },
        h("span", { class: "tx-corner-top" }, "Search procedure", h("span", { class: "tx-arrow" }, "→")),
        h("span", { class: "tx-corner-bottom" }, h("span", { class: "tx-arrow" }, "↓"), "Homogeneity criterion"),
      ),
      1,
      2,
    );

    // column headers
    for (const key of ORDER) {
      const s = SEARCH[key];
      const el = h(
        "button",
        {
          type: "button",
          class: `tx-colhead is-${s.task}`,
          "aria-pressed": "false",
          title: "Show what this search procedure does",
          onclick: () => setFocus(`search:${key}`),
        },
        h("span", { class: "tx-colhead-name" }, s.name),
        h("span", { class: "tx-mf", html: MF[key] }),
        h("span", { class: "tx-colhead-desc" }, s.short),
      );
      colHeads[key] = el;
      place(el, COL[key], 2);
    }

    // row headers
    for (const c of D.criteria) {
      const el = h(
        "button",
        {
          type: "button",
          class: "tx-rowhead",
          "aria-pressed": "false",
          title: "Show what this criterion compares",
          onclick: () => setFocus(`row:${c.key}`),
        },
        h("span", { class: "tx-rowhead-name" }, c.name),
        h("span", { class: "tx-mf", html: RF[c.key] }),
        h("span", { class: "tx-rowhead-desc" }, c.short),
      );
      rowHeads[c.key] = el;
      place(el, 1, ROW[c.key]);
    }

    // cells
    for (const key of ORDER) {
      const barMembers = METHODS.filter((m) => m.search === key && m.bar);
      for (const c of D.criteria) {
        const inCell = METHODS.filter((m) => m.search === key && !m.bar && m.rows[0] === c.key);
        inCell.sort((a, b) => (isTsseg(b) ? 1 : 0) - (isTsseg(a) ? 1 : 0));
        const cell = h("div", { class: `tx-cell is-${SEARCH[key].task}`, dataset: { search: key, row: c.key } });
        const costIdx = D.costs.findIndex((x) => x.row === c.key);
        if (barMembers.length && costIdx > 0) {
          const piece = h(
            "div",
            { class: "tx-bar tx-bar-bottom", "aria-hidden": "true" },
            h("span", { class: "tx-bar-cost" }, h("i", {}, "cost: "), D.costs[costIdx].label),
          );
          cell.append(piece);
          bars.push([[piece], barMembers]);
        }
        cell.append(h("div", { class: "tx-chips" }, inCell.map((m) => chip(m))));
        if (barMembers.length && costIdx === 0) {
          const piece = h(
            "div",
            { class: "tx-bar tx-bar-top", role: "group", "aria-label": "Cost chosen by the user (model=)" },
            h("div", { class: "tx-bar-chips" }, barMembers.map((m) => chip(m, true))),
            h("span", { class: "tx-bar-cost" }, h("i", {}, "cost: "), D.costs[0].label),
          );
          cell.append(piece);
          bars.push([[piece], barMembers]);
        }
        cells.push([cell, key, c.key]);
        place(cell, COL[key], ROW[c.key]);
      }
    }

    // DP note
    place(h("div", { class: "tx-dpnote", html: D.dpNote }), COL.optimal, LAST, COL.states - COL.optimal + 1);

    return h("div", { class: "tx-grid-scroll", tabindex: "0", "aria-label": "Taxonomy grid, scrolls horizontally" }, grid);
  }

  // ------------------------------------------------------------------- list
  const listRows = []; // [tr, method]
  const COLUMNS = [
    { key: "name", label: "Method", get: (m) => m.name.toLowerCase() },
    { key: "task", label: "Task", get: (m) => m.task },
    { key: "search", label: "Search procedure", get: (m) => ORDER.indexOf(m.search) },
    { key: "row", label: "Criterion", get: (m) => D.criteria.findIndex((c) => c.key === m.rows[0]) },
    { key: "modes", label: "Modes", get: (m) => (m.unsup ? 0 : 2) + (m.guided ? 0 : 1) },
    { key: "cx", label: "Complexity", get: () => 0, sortable: false },
    { key: "install", label: "Install", get: (m) => (m.extras || []).join(",") },
    { key: "ref", label: "Reference", get: (m) => m.ref || "" },
  ];
  let tbody;
  const sortButtons = {};
  function buildList() {
    const head = h(
      "tr",
      {},
      COLUMNS.map((c) => {
        if (c.sortable === false) return h("th", { scope: "col" }, c.label);
        const b = h(
          "button",
          {
            type: "button",
            class: "tx-sort",
            onclick: () => {
              state.sort = { key: c.key, dir: state.sort.key === c.key ? -state.sort.dir : 1 };
              sortList();
            },
          },
          c.label,
          h("span", { class: "tx-sort-ind", "aria-hidden": "true" }),
        );
        sortButtons[c.key] = b;
        return h("th", { scope: "col" }, b);
      }),
    );
    tbody = h("tbody");
    for (const m of METHODS) {
      const lit = !isTsseg(m);
      const modes = lit
        ? h("span", { class: "tx-muted" }, "–")
        : h(
            "span",
            { class: "tx-modes" },
            m.unsup ? h("span", { class: "tx-tag" }, "unsupervised") : null,
            m.guided
              ? h("span", { class: `tx-tag ${m.unsup ? "" : "is-req"}` }, m.unsup ? "guided" : "guided only")
              : null,
          );
      const tr = h(
        "tr",
        { class: lit ? "is-lit" : m.status === "deprecated" ? "is-depr" : "" },
        h(
          "th",
          { scope: "row" },
          lit
            ? h("span", { class: "tx-list-name" }, m.name, h("span", { class: "tx-chip-tag" }, "not in tsseg"))
            : h(
                "a",
                { href: m.url, class: "tx-list-name" },
                h("span", { class: `tx-dot tx-dot-${m.task}` }),
                m.name,
                m.status === "deprecated" ? h("span", { class: "tx-chip-tag" }, "deprecated") : null,
              ),
          lit ? null : h("code", { class: "tx-list-cls" }, m.cls),
        ),
        h("td", {}, TASK_SHORT[m.task]),
        h("td", {}, SEARCH[m.search].name),
        h("td", {}, critLabel(m)),
        h("td", {}, modes),
        h("td", { class: "tx-list-cx", html: m.cx ? m.cx + (m.typical ? "*" : "") : "" }),
        h(
          "td",
          {},
          lit
            ? ""
            : m.extras.length
              ? h("code", {}, `tsseg[${m.extras.join(",")}]`)
              : h("span", { class: "tx-muted" }, "core"),
        ),
        h("td", { class: "tx-list-ref" }, m.ref || ""),
      );
      listRows.push([tr, m]);
    }
    sortList();
    return h(
      "div",
      { class: "tx-list-scroll" },
      h("table", { class: "tx-list" }, h("thead", {}, head), tbody),
    );
  }
  function sortList() {
    const col = COLUMNS.find((c) => c.key === state.sort.key);
    const rows = [...listRows].sort((a, b) => {
      const ka = col.get(a[1]);
      const kb = col.get(b[1]);
      const lit = (isTsseg(a[1]) ? 0 : 1) - (isTsseg(b[1]) ? 0 : 1);
      if (lit) return lit;
      if (ka < kb) return -state.sort.dir;
      if (ka > kb) return state.sort.dir;
      return a[1].name.localeCompare(b[1].name);
    });
    tbody.replaceChildren(...rows.map((r) => r[0]));
    for (const [key, b] of Object.entries(sortButtons)) {
      const th = b.parentElement;
      if (key === state.sort.key) th.setAttribute("aria-sort", state.sort.dir > 0 ? "ascending" : "descending");
      else th.removeAttribute("aria-sort");
    }
  }

  // ------------------------------------------------------------------- card
  const card = h("div", { class: "tx-card", role: "tooltip", id: "tx-card", hidden: true });
  let hot = [];
  function fact(label, value) {
    return h("div", { class: "tx-fact" }, h("dt", {}, label), h("dd", {}, value));
  }
  function showCard(m, el) {
    const lit = !isTsseg(m);
    card.replaceChildren(
      h(
        "div",
        { class: "tx-card-top" },
        h("span", { class: `tx-pill is-${m.task}` }, TASK[m.task]),
        lit ? h("span", { class: "tx-pill is-lit" }, "not in tsseg") : null,
        m.status === "deprecated" ? h("span", { class: "tx-pill is-depr" }, "deprecated") : null,
        m.bench ? h("span", { class: "tx-pill is-bench" }, "TSB-SEG") : null,
      ),
      h("div", { class: "tx-card-name" }, m.name),
      lit ? null : h("code", { class: "tx-card-cls" }, m.cls),
      m.approach ? h("p", { class: "tx-card-approach" }, m.approach) : null,
      h(
        "dl",
        { class: "tx-card-facts" },
        fact("Search", SEARCH[m.search].name),
        fact("Criterion", critLabel(m)),
        m.cx ? fact("Complexity", h("span", { class: "tx-cx", html: m.cx + (m.typical ? "*" : "") })) : null,
        lit
          ? null
          : fact(
              "Modes",
              [m.unsup ? "unsupervised" : null, m.guided ? (m.unsup ? "guided" : "guided only") : null]
                .filter(Boolean)
                .join(" · "),
            ),
        lit ? null : fact("Channels", m.mv ? (m.uni ? "univariate and multivariate" : "multivariate only") : "univariate only"),
        lit
          ? null
          : fact(
              "Install",
              m.extras.length ? h("code", {}, `pip install "tsseg[${m.extras.join(",")}]"`) : "core: pip install tsseg",
            ),
      ),
      m.ref ? h("p", { class: "tx-card-ref" }, m.ref) : null,
      lit ? null : h("div", { class: "tx-card-cta" }, "Open the documentation", h("span", { class: "tx-icon", html: ICON.arrow })),
    );
    card.hidden = false;
    el.setAttribute("aria-describedby", "tx-card");
    // position next to the chip, inside the viewport
    const r = el.getBoundingClientRect();
    const cw = card.offsetWidth;
    const ch = card.offsetHeight;
    let x = r.right + 12;
    if (x + cw > window.innerWidth - 12) x = r.left - cw - 12;
    if (x < 12) x = Math.min(Math.max(12, r.left), window.innerWidth - cw - 12);
    let y = r.top - 8;
    if (x === Math.min(Math.max(12, r.left), window.innerWidth - cw - 12)) y = r.bottom + 10;
    y = Math.max(12, Math.min(y, window.innerHeight - ch - 12));
    card.style.left = `${x}px`;
    card.style.top = `${y}px`;
    // highlight the column and row headers of the method
    hot = [colHeads[m.search], ...m.rows.map((r) => rowHeads[r])].filter(Boolean);
    hot.forEach((x) => x.classList.add("is-hot"));
    for (const [els, members] of bars) if (members.includes(m)) els.forEach((x) => x.classList.add("is-hot"));
  }
  function hideCard(e) {
    card.hidden = true;
    if (e && e.currentTarget) e.currentTarget.removeAttribute("aria-describedby");
    hot.forEach((x) => x.classList.remove("is-hot"));
    hot = [];
    app.querySelectorAll(".tx-bar.is-hot").forEach((x) => x.classList.remove("is-hot"));
  }
  window.addEventListener("scroll", () => hideCard(), { passive: true });

  // ----------------------------------------------------------------- legend
  function legend() {
    const sample = (task, name, extra) =>
      h("span", { class: `tx-chip is-${task} is-sample ${extra || ""}` }, h("span", { class: "tx-chip-name" }, name));
    return h(
      "div",
      { class: "tx-legend" },
      h(
        "div",
        { class: "tx-legend-items" },
        h("span", { class: "tx-legend-item" }, sample("cpd", "PELT"), sample("sd", "CLaP"), "tsseg detector: click to open its page"),
        h("span", { class: "tx-legend-item" }, sample("cpd", "gSeg", "is-lit"), "in the literature, not in tsseg"),
        h("span", { class: "tx-legend-item" }, h("span", { class: "tx-k tx-k-req" }, "K"), "needs the number of change points or states"),
        h("span", { class: "tx-legend-item" }, h("span", { class: "tx-k tx-k-opt" }, "K"), "accepts it (optional)"),
        h("span", { class: "tx-legend-item" }, h("span", { class: "tx-badge-pkg", html: ICON.pkg }), "needs an optional extra"),
        h(
          "span",
          { class: "tx-legend-item" },
          h("span", { class: "tx-bar tx-bar-sample" }, h("i", {}, "cost: "), "rbf"),
          "one search, cost chosen by the user (model=): the box spans the rows of its costs",
        ),
      ),
      h(
        "p",
        { class: "tx-legend-note" },
        h("b", {}, "Complexity"),
        " of one run as implemented, default settings, ℓ₂ cost; epochs and sampler or EM iterations omitted. ",
        h("i", {}, "n"),
        " length, ",
        h("i", {}, "d"),
        " channels, ",
        h("i", {}, "n"),
        h("sub", {}, "CP"),
        " change points, ",
        h("i", {}, "n"),
        h("sub", {}, "S"),
        " states, ℓ window length, ",
        h("i", {}, "b"),
        " kernel basis size; * typical case, quadratic at worst.",
      ),
      h(
        "p",
        { class: "tx-legend-note" },
        "Not in the grid: ",
        D.excluded.map((x, i) => [i ? ", " : "", h("a", { href: x.url }, h("code", {}, x.cls)), ` (${x.why})`]),
        ". The axes follow the taxonomy of the ",
        h("a", { href: D.tsbseg }, "TSB-SEG benchmark"),
        ".",
      ),
    );
  }

  // ----------------------------------------------------------------- update
  const gridView = buildGrid();
  const listView = buildList();
  const empty = h(
    "div",
    { class: "tx-empty", hidden: true },
    h("strong", {}, "No detector matches."),
    " Try fewer filters, or ",
    h(
      "button",
      { type: "button", class: "tx-linkbtn", onclick: () => reset.click() },
      "reset everything",
    ),
    ".",
  );
  const stage = h("div", { class: "tx-stage" }, gridView, listView, empty);

  function currentView() {
    return state.view || (window.matchMedia("(max-width: 900px)").matches ? "list" : "grid");
  }

  function update() {
    const view = currentView();
    gridView.hidden = view !== "grid";
    listView.hidden = view !== "list";

    // chips
    let shown = 0;
    let total = 0;
    let litShown = 0;
    const st = new Map(METHODS.map((m) => [m, status(m)]));
    for (const [el, m] of chipEls) {
      const s = st.get(m);
      el.classList.toggle("is-hidden", s === "hide");
      el.classList.toggle("is-dim", s === "dim");
      el.tabIndex = s === "on" ? 0 : -1;
    }
    for (const m of METHODS) {
      const s = st.get(m);
      if (!isTsseg(m)) {
        if (s === "on") litShown++;
        continue;
      }
      if (s === "hide") continue;
      total++;
      if (s === "on") shown++;
    }
    for (const [els, members] of bars) {
      const anyOn = members.some((m) => st.get(m) === "on");
      els.forEach((x) => x.classList.toggle("is-dim", !anyOn));
    }
    // focus (selected column or row)
    const [fk, fv] = (state.focus || ":").split(":");
    for (const [cell, s, r] of cells) {
      const inFocus = !state.focus || (fk === "search" ? s === fv : r === fv);
      cell.classList.toggle("is-unfocused", !inFocus);
      const any = METHODS.some(
        (m) => m.search === s && (m.bar ? m.rows.includes(r) : m.rows[0] === r) && st.get(m) === "on",
      );
      cell.classList.toggle("is-quiet", !any);
    }
    for (const [k, el] of Object.entries(colHeads)) {
      el.setAttribute("aria-pressed", String(fk === "search" && fv === k));
      el.classList.toggle("is-dim", state.task !== "all" && SEARCH[k].task !== state.task);
    }
    for (const [k, el] of Object.entries(rowHeads)) el.setAttribute("aria-pressed", String(fk === "row" && fv === k));
    app.classList.toggle("has-focus", Boolean(state.focus));

    // list rows
    for (const [tr, m] of listRows) {
      const s = st.get(m);
      const inFocus = !state.focus || (fk === "search" ? m.search === fv : m.rows.includes(fv));
      tr.hidden = s !== "on" || !inFocus;
    }

    // toolbar
    taskSeg.sync();
    viewSeg.sync();
    litToggle.sync();
    deprToggle.sync();
    filterButtons.forEach((b) => b.setAttribute("aria-pressed", String(state.filters.has(b.dataset.key))));
    const dirty = state.q || state.task !== "all" || state.filters.size || state.lit || state.depr || state.focus;
    reset.hidden = !dirty;
    count.replaceChildren(
      h("b", {}, String(shown)),
      shown === total ? ` detector${shown === 1 ? "" : "s"}` : ` of ${total} detectors`,
      state.lit ? ` · ${litShown} from the literature` : "",
    );
    empty.hidden = shown + litShown > 0;
    renderFocus();
    writeHash();
  }

  // ------------------------------------------------------------------ input
  let timer;
  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      state.q = input.value.trim();
      update();
    }, 80);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      input.value = "";
      state.q = "";
      update();
    } else if (e.key === "Enter") {
      // open the only match
      const on = METHODS.filter((m) => isTsseg(m) && status(m) === "on");
      if (on.length === 1) window.location.href = on[0].url;
    }
  });
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    const typing = t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
    if (e.key === "/" && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
      e.preventDefault();
      input.focus();
      input.select();
    } else if (e.key === "Escape" && !typing && state.focus) {
      state.focus = null;
      update();
    }
  });
  window.addEventListener("hashchange", () => {
    readHash();
    input.value = state.q;
    update();
  });
  window.matchMedia("(max-width: 900px)").addEventListener("change", () => {
    if (!state.view) update();
  });

  // ------------------------------------------------------------------ mount
  readHash();
  input.value = state.q;
  app.replaceChildren(toolbar, focusPanel, stage, legend());
  document.body.append(card);
  app.dataset.state = "ready";
  update();
  spotlight();

  function spotlight() {
    if (!state.spot) return;
    const m = BY_ID[state.spot];
    state.spot = null;
    if (m.status === "deprecated" && !state.depr) {
      state.depr = true;
      update();
    }
    const entry = chipEls.find(([, x]) => x === m);
    if (!entry || currentView() !== "grid") return;
    const el = entry[0];
    requestAnimationFrame(() => {
      el.scrollIntoView({ block: "center", inline: "center" });
      el.classList.add("is-spot");
      setTimeout(() => el.focus({ preventScroll: true }), 350);
      setTimeout(() => el.classList.remove("is-spot"), 4000);
    });
  }
})();
