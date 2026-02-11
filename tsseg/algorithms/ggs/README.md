# GGS (Greedy Gaussian Segmentation)

Adapted from the [cvxgrp/GGS](https://github.com/cvxgrp/GGS) repository
(Stanford Convex Optimization Group). GGS models each segment as a
multivariate Gaussian and greedily splits the time series to maximise the
total log-likelihood improvement, with a covariance-regularisation penalty.

## Source

Adapted from [cvxgrp/GGS](https://github.com/cvxgrp/GGS) (BSD 2-Clause
license, see `LICENSE` in this directory).

## References

```bibtex
@article{hallac2019greedy,
  title   = {Greedy {G}aussian Segmentation of Multivariate Time Series},
  author  = {Hallac, David and Vare, Sagar and Boyd, Stephen and Leskovec, Jure},
  journal = {Advances in Data Analysis and Classification},
  volume  = {13},
  number  = {3},
  pages   = {727--751},
  year    = {2019},
  doi     = {10.1007/s11634-018-0335-0}
}
```
