# Window (Window-Based Segmentation)

Sliding-window change point detection. Computes a discrepancy (gain) score by
comparing the distributions inside two adjacent windows as they slide along the
signal. Peaks in the score profile indicate change points.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised (`penalty` or `epsilon`)
- Supports multiple cost models: L2, L1, RBF, linear, normal, cosine
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures `Window` solver.

- Origin: vendored from ruptures v1.1.8
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@book{basseville1993detection,
  title     = {Detection of Abrupt Changes: Theory and Application},
  author    = {Basseville, Mich{\`e}le and Nikiforov, Igor V.},
  year      = {1993},
  publisher = {Prentice Hall}
}
```
