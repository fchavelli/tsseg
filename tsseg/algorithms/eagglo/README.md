# E-Agglo (E-Agglomerative)

Non-parametric hierarchical agglomerative algorithm for multiple change point
analysis. Merges adjacent clusters greedily to maximise a goodness-of-fit
statistic, the sum of the energy divergences between adjacent clusters, and
keeps the segmentation with the largest statistic along the merging sequence.

## Key properties

- Type: change point detection
- Unsupervised: the number of change points maximises the (optionally
  penalised) goodness-of-fit statistic; from one cluster per point it
  usually over-segments, see the class docstring
- Non-parametric
- Univariate and multivariate
- O(n^2) time; O(m^2) memory for m initial clusters (O(n^2) by default)
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
