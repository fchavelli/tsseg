# GGS (Greedy Gaussian Segmentation)

Models each segment as a multivariate Gaussian distribution. Greedily selects
split points that maximise the total log-likelihood, then optionally refines
them with a local search.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`)
- Multivariate (models full covariance structure)
- Greedy with optional refinement

## Implementation

Adapted from the Stanford Convex Optimization Group's GGS repository.

- Origin: adapted from https://github.com/cvxgrp/GGS
- Licence: BSD 2-Clause (Copyright (c) 2018, Stanford University Convex Optimization Group)
- Licence file: `LICENSE` in this directory

Difference from the reference code: in the split search
(`add_new_change_point`), the running mean of the left part starts from the
first sample, `data[0]`. The reference code (`ggs.py`) starts it from
`data[0] / n`, `n` being the number of channels, which is right only for a
univariate series: every multivariate split was scored with a wrong left
covariance, and the change points depended on a constant offset of the series.
Univariate results are unchanged.

## Citation

```bibtex
@article{hallac2019greedy,
  title   = {Greedy Gaussian Segmentation of Multivariate Time Series},
  author  = {Hallac, David and Nystrup, Peter and Boyd, Stephen},
  journal = {Advances in Data Analysis and Classification},
  volume  = {13},
  pages   = {727--751},
  year    = {2019}
}
```
