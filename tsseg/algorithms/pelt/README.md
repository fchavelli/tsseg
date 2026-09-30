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
- Change: pruning is delayed by `min_size` (fix of upstream
  [ruptures PR #383](https://github.com/deepcharles/ruptures/pull/383)); before
  it, PELT could return a sub-optimal segmentation when `min_size > 1`
- Costs `l1`, `l2`, `rbf`, `cosine`: with numba installed (`backend="auto"`),
  the whole search runs in numba (`ruptures/accel/`). For the kernel costs it is
  a transcription of the C code of ruptures' `KernelCPD`, which updates the
  kernel sums sample by sample in O(n) memory instead of reading them off the
  n x n Gram matrix; for `l1` and `l2`, the costs of the segments that end at t
  come from one sweep back from t. It returns the same segmentation as the
  Python path (`backend="python"`).
- Ties: candidates closer than 1e-9 of the costs involved are tied, and the
  earliest change point wins, on both backends.
- Penalty: `pen_scale="bic"` makes `penalty` a coefficient on `log(n) * u`, u the
  cost of one sample of the whole signal (`ruptures.utils.bic_penalty`): d for
  `l2` on unit-variance channels, where it is the BIC penalty, 1 for the kernel
  costs. One value then applies across lengths, dimensions and costs.
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

For a univariate signal with `model="l1"`, upstream ruptures now ships
[`L1Potts`](https://github.com/deepcharles/ruptures/blob/master/src/ruptures/detection/l1potts.py)
(Storath, Weinmann & Unser, 2017), an exact and much faster solver of the same
penalised problem with `min_size=1` and `jump=1`. It is not vendored in tsseg.

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
