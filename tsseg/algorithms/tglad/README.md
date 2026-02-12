# tGLAD

Change point detection by comparing successive Graphical Lasso precision
matrices estimated on sliding windows. A significant change in the graph
structure (measured by Frobenius distance between precision matrices) indicates
a regime transition.

## Key properties

- Type: change point detection
- Unsupervised (threshold-based)
- Univariate and multivariate
- Requires PyTorch and networkx

## Implementation

Wraps the uGLAD multitask solver, which is vendored under `vendor/uGLAD/`. The
`vendor/__init__.py` module registers the vendored package into `sys.modules` so
that existing absolute imports work without modification.

- Origin: wrapper around tGLAD / uGLAD
- Source: https://github.com/Harshs27/tGLAD
- Licence: uGLAD Non-Commercial License (Copyright (c) 2023, Harsh Shrivastava)
- Licence file: `vendor/LICENSE`

## Citation

```bibtex
@article{shrivastava2022tglad,
  title   = {{tGLAD}: A Sparse Graph Recovery based Approach for
             Multivariate Time Series Segmentation},
  author  = {Shrivastava, Harsh and others},
  year    = {2022}
}
```
