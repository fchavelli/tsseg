# E-Agglo

**E-Agglo** (E-Agglomerative) is a non-parametric, hierarchical agglomerative
algorithm for multiple change point analysis of multivariate data.

The method starts by assigning each observation to its own cluster, then
greedily merges adjacent clusters to maximise a goodness-of-fit statistic based
on the $\alpha$-th absolute moment of pairwise Euclidean distances.  The merge
sequence that yields the highest GoF determines both the number and the
locations of change points — no prior knowledge of the number of segments is
required.

**Key properties:**

- Fully unsupervised (automatic selection of the number of change points)
- Non-parametric (no distributional assumption beyond existence of the
  $\alpha$-th moment, $\alpha \in (0, 2]$)
- Preserves temporal ordering of observations
- $O(n^2)$ time and memory — practical only for short to moderate length series

---

## Implementation

This code is adapted from the **aeon** toolkit:

> Source: <https://github.com/aeon-toolkit/aeon/blob/v1.3.0/aeon/segmentation/_eagglo.py>
>
> License: **BSD 3-Clause License** — Copyright (c) aeon developers.

which itself is a Python port of the R package **ecp**:

> <https://github.com/cran/ecp/blob/master/R/e_agglomerative.R>

---

## Citation

```bibtex
@article{matteson2014nonparametric,
    title     = {A Nonparametric Approach for Multiple Change Point Analysis
                 of Multivariate Data},
    author    = {Matteson, David S. and James, Nicholas A.},
    journal   = {Journal of the American Statistical Association},
    volume    = {109},
    number    = {505},
    pages     = {334--345},
    year      = {2014},
    doi       = {10.1080/01621459.2013.849605},
}
```
