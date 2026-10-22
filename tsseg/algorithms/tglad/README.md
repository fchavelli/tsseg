# tGLAD

Change point detection by comparing the conditional independence graphs that
uGLAD (unrolled graphical lasso) recovers on sliding windows. The score of a
window is the absolute second difference, across windows, of the sum of its
partial correlations, as in `main.py` of the authors' repository; a score above
`threshold` marks a regime transition.

## Key properties

- Type: change point detection
- Unsupervised (threshold-based)
- Univariate and multivariate
- Requires PyTorch and networkx

## Implementation

Wraps the uGLAD multitask solver, which is vendored under `vendor/uGLAD/` (a
copy of the `uGLAD/` folder of the tGLAD repository, imports made relative). The
`vendor/__init__.py` module registers the vendored package into `sys.modules` so
that existing absolute imports work without modification.

- Origin: wrapper around tGLAD / uGLAD
- Source: https://github.com/Harshs27/tGLAD
- Licence: uGLAD Non-Commercial License (Copyright (c) 2023, Harsh Shrivastava)
- Licence file: `vendor/LICENSE`

## Citation

```bibtex
@article{imani2023uglad,
  title   = {Are {uGLAD}? Time will tell!},
  author  = {Imani, Shima and Shrivastava, Harsh},
  journal = {arXiv preprint arXiv:2303.11647},
  year    = {2023}
}
```
