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
- Costs `l1`, `l2`, `rbf`, `cosine`: with numba installed (`backend="auto"`),
  the segment costs are computed in numba (`ruptures/accel/`): the kernel costs
  without the n x n Gram matrix, `l1` and `l2` bit for bit as numpy computes
  them. It returns the same change points as the Python path
  (`backend="python"`).
- Ties: merge gains closer than 1e-9 of the segments' costs are tied; the
  leftmost merge is done first, as upstream among exact ties.
- Penalty: `pen_scale="bic"` makes `penalty` a coefficient on `log(n) * u`, u the
  cost of one sample of the whole signal (`ruptures.utils.bic_penalty`): d for
  `l2` on unit-variance channels, where it is the BIC penalty, 1 for the kernel
  costs. One value then applies across lengths, dimensions and costs.
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
