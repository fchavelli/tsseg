# AMOC

**AMOC** (At Most One Change) is the simplest change point detector: it
searches for the single breakpoint that minimises the total sum of squared
errors (SSE) on either side of the split.

$$\hat{\tau} = \arg\min_{\tau} \left[
    \sum_{t=1}^{\tau} \|x_t - \bar{x}_{1:\tau}\|^2
  + \sum_{t=\tau+1}^{T} \|x_t - \bar{x}_{\tau+1:T}\|^2
\right]$$

It is a foundational building block for multi-change detectors such as Binary
Segmentation and PELT, which repeatedly apply the single-change solver on
sub-segments.

**Key properties:**

- Fully unsupervised (no hyperparameter for the number of segments)
- Detects **at most one** change point per call
- Works on univariate and multivariate series
- $O(n \cdot d)$ time, $O(1)$ extra memory (beyond the input)

---

## Implementation

This is an independent reimplementation of the AMOC objective.  The API design
and references are inspired by the R **changepoint** package:

> Source: <https://github.com/rkillick/changepoint/>
>
> CRAN: <https://cran.r-project.org/web/packages/changepoint/>
>
> License: **GPL (>= 2)**

The present Python code does not reuse any R source code; it is a clean-room
implementation of the classical SSE-based single change point criterion.

---

## Citation

```bibtex
@article{killick2012optimal,
    title     = {Optimal Detection of Changepoints with a Linear Computational
                 Cost},
    author    = {Killick, Rebecca and Fearnhead, Paul and Eckley, Idris A.},
    journal   = {Journal of the American Statistical Association},
    volume    = {107},
    number    = {500},
    pages     = {1590--1598},
    year      = {2012},
    doi       = {10.1080/01621459.2012.737745},
}
```
