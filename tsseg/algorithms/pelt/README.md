# PELT (Pruned Exact Linear Time)

Exact change point detection with an optimal pruning rule that reduces the
average-case complexity to O(n). Based on dynamic programming with a penalty
term; the pruning discards candidate split points that are provably
sub-optimal.

## Key properties

- Type: change point detection
- Fully unsupervised (penalty-based)
- Supports multiple cost models: L2, L1, RBF, linear, normal, cosine
- O(n) average time
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures `Pelt` solver.

- Origin: vendored from ruptures v1.1.8
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@article{killick2012optimal,
  title   = {Optimal Detection of Changepoints with a Linear Computational Cost},
  author  = {Killick, Rebecca and Fearnhead, Paul and Eckley, Idris A.},
  journal = {Journal of the American Statistical Association},
  volume  = {107},
  number  = {500},
  pages   = {1590--1598},
  year    = {2012}
}
```
