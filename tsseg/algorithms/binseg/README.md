# BinSeg (Binary Segmentation)

One of the oldest and most widely used change point detection algorithms.
Recursively applies a single-change test and splits the signal at the most
significant change point until a stopping criterion is met.

## Key properties

- Type: change point detection
- Semi-supervised (provide `n_cps`) or unsupervised (provide `penalty` or `epsilon`)
- Supports multiple cost models: L2, L1, RBF, linear, normal, cosine
- O(n log n) time in practice
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures `Binseg` solver.

- Origin: vendored from ruptures v1.1.8
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@article{truong2020selective,
  title   = {Selective Review of Offline Change Point Detection Methods},
  author  = {Truong, Charles and Oudre, Laurent and Vayatis, Nicolas},
  journal = {Signal Processing},
  volume  = {167},
  pages   = {107299},
  year    = {2020}
}
```
