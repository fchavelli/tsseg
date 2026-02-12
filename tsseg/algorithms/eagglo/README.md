# E-Agglo (E-Agglomerative)

Non-parametric hierarchical agglomerative algorithm for multiple change point
analysis. Maximises a goodness-of-fit statistic based on the alpha-th absolute
moment of pairwise Euclidean distances by iteratively merging adjacent clusters.

## Key properties

- Type: change point detection
- Fully unsupervised (penalty-based stopping)
- Non-parametric
- Univariate and multivariate
- O(n^2) time and memory
- Uses numba for acceleration

## Implementation

Adapted from the aeon toolkit, which itself is a Python port of the R `ecp`
package.

- Origin: adapted from aeon
- Licence: BSD 3-Clause (aeon toolkit)

## Citation

```bibtex
@article{matteson2014nonparametric,
  title   = {A Nonparametric Approach for Multiple Change Point Analysis of
             Multivariate Data},
  author  = {Matteson, David S. and James, Nicholas A.},
  journal = {Journal of the American Statistical Association},
  volume  = {109},
  number  = {505},
  pages   = {334--345},
  year    = {2014}
}
```
