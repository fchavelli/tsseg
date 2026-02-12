# BottomUp (Bottom-Up Segmentation)

Agglomerative segmentation algorithm. Starts from the finest possible
segmentation and iteratively merges the pair of adjacent segments with the
lowest cost increase until a stopping criterion is reached.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised (`penalty` or `epsilon`)
- Supports multiple cost models: L2, L1, RBF, linear, normal, cosine
- O(n log n) time
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures `BottomUp` solver.

- Origin: vendored from ruptures v1.1.8
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@article{keogh2001online,
  title   = {An Online Algorithm for Segmenting Time Series},
  author  = {Keogh, Eamonn and Chu, Selina and Hart, David and Pazzani, Michael},
  journal = {Proceedings of the IEEE International Conference on Data Mining},
  pages   = {289--296},
  year    = {2001}
}
```
