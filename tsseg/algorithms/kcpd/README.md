# KCPD (Kernel Change Point Detection)

> **Deprecated.** `KCPDDetector` is `PeltDetector` (with `pen`) or `DynpDetector`
> (with `n_cps`) on a kernel cost, and will be removed in a future release. Use
> `PeltDetector(model="rbf", penalty=pen, jump=1)` or
> `DynpDetector(model="rbf", n_cps=k, jump=1)` (`model="l2"` for the linear
> kernel); decimate the input yourself for `decimation`.

Kernel-based change point detection. Operates in a Reproducing Kernel Hilbert
Space and uses either dynamic programming (exact) or a penalised formulation to
locate change points based on the maximum mean discrepancy between segments.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised (`penalty`)
- Supports RBF, linear and cosine kernels
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures kernel-based cost and detection pipeline: the same
solvers as `PeltDetector` (with `pen`) and `DynpDetector` (with `n_cps`) on a
kernel cost (`linear` is the `l2` cost), plus `decimation` and `pen_scale`. With
numba, as in those detectors, the search runs in numba and the `rbf` and
`cosine` kernels in O(n) memory.

- Origin: vendored from ruptures v1.1.8
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@article{arlot2019kernel,
  title   = {A Kernel Multiple Change-point Algorithm via Model Selection},
  author  = {Arlot, Sylvain and Celisse, Alain and Harchaoui, Zaid},
  journal = {Journal of Machine Learning Research},
  volume  = {20},
  number  = {162},
  pages   = {1--56},
  year    = {2019}
}
```
