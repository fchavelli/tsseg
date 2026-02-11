# BinSeg

**Binary Segmentation** (BinSeg) is one of the oldest and most widely used
change point detection algorithms.  It works by recursively applying a
single-change test (e.g. AMOC / CUSUM) and splitting the signal at the most
significant breakpoint until a stopping criterion is met.

**Key properties:**

- Semi-supervised (provide `n_cps`) or unsupervised (provide `penalty` or
  `epsilon` for automatic selection)
- Supports multiple cost models: L2 (mean shift), L1, RBF, etc.
- $O(n \log n)$ time in practice (greedy top-down search)
- Univariate and multivariate

---

## Implementation

The active detector (`detector.py`) wraps the **ruptures** Binary Segmentation
solver, re-implemented locally under `tsseg.algorithms.ruptures`:

> Original ruptures library: <https://github.com/deepcharles/ruptures>
>
> License: **BSD 2-Clause License** — Copyright (c) 2017, ENS Paris-Saclay,
> CNRS.

---

## References

```bibtex
@article{bai1997estimating,
    title     = {Estimating Multiple Breaks One at a Time},
    author    = {Bai, Jushan},
    journal   = {Econometric Theory},
    volume    = {13},
    number    = {3},
    pages     = {315--352},
    year      = {1997},
    doi       = {10.1017/S0266466600005831},
}

@article{fryzlewicz2014wild,
    title     = {Wild Binary Segmentation for Multiple Change-Point Detection},
    author    = {Fryzlewicz, Piotr},
    journal   = {The Annals of Statistics},
    volume    = {42},
    number    = {6},
    pages     = {2243--2281},
    year      = {2014},
    doi       = {10.1214/14-AOS1245},
}
```
