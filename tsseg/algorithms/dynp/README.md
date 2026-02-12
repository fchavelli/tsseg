# DynP (Dynamic Programming Segmentation)

Exact optimal partitioning algorithm. Given a fixed number of change points,
finds the segmentation that globally minimises the sum of segment costs.

## Key properties

- Type: change point detection
- Semi-supervised only (requires `n_cps`)
- Exact (globally optimal for the given cost)
- Supports multiple cost models: L2, L1, RBF, linear, normal, cosine
- O(C Q n^2) time, with C the number of change points, Q the complexity of the cost function on one sub-series
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures `Dynp` solver.

- Origin: vendored from ruptures v1.1.8
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@article{auger1989algorithms,
  title   = {Algorithms for the Optimal Identification of Segment Neighborhoods},
  author  = {Auger, Ivan E. and Lawrence, Charles E.},
  journal = {Bulletin of Mathematical Biology},
  volume  = {51},
  number  = {1},
  pages   = {39--54},
  year    = {1989}
}
```
