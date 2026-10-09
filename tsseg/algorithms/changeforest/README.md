# changeforest (random forests for change point detection)

Classifier-based, nonparametric multiple change point detection. To score a
split of a segment, changeforest trains a classifier (a random forest by
default) to separate the observations before the split from those after it,
and turns the out-of-bag class probabilities into a classifier log-likelihood
ratio, the gain. A two-step search fits the classifier at the 1/4, 1/2 and 3/4
quantiles of the segment, maximises the approximate gain curves, then refits
the classifier at the best guess and maximises the new gain curve over all
candidate splits; a pseudo-permutation test decides whether the best split is
kept, and binary segmentation recurses on kept splits.

## Key properties

- Type: change point detection
- Unsupervised (permutation test) or semi-supervised (`n_cps`, best-first
  binary segmentation on the package's gains; not part of the paper)
- Univariate and multivariate; detects changes in the whole distribution
  (mean, covariance, shape) without a parametric model
- Gains: random forest (default), k-nearest neighbours, change in mean; only
  the random forest is invariant to the scale of each channel
- Search: binary (the paper's method), seeded binary or wild binary
  segmentation; the paper's model selection is designed for binary
  segmentation and over-segments with the other two
- Assumes independent observations: over-segments autocorrelated series

## Implementation

Wraps the official `changeforest` package from PyPI
(`pip install tsseg[changeforest]`, Rust with Python bindings); no code is
vendored. Defaults are those of the package (`Control` in version 1.2.1), which
match the settings of the paper's main simulations.

- Origin: changeforest Python package (Malte Londschien)
- Source: https://github.com/mlondschien/changeforest
- Licence: BSD-3-Clause. tsseg only depends on it.

## Citation

```bibtex
@article{londschien2023random,
  title={Random forests for change point detection},
  author={Londschien, Malte and B{\"u}hlmann, Peter and Kov{\'a}cs, Solt},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={216},
  pages={1--45},
  year={2023}
}
```
