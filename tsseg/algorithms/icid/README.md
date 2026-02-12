# iCID (Isolation Distributional Kernel Change Interval Detection)

Change point detection based on isolation distributional kernels. Transforms
the time series through an aNNEspace embedding, computes a dissimilarity score
between consecutive windows, automatically selects the isolation parameter psi,
and applies adaptive thresholding to emit change points.

## Key properties

- Type: change point detection
- Fully unsupervised (automatic psi selection and adaptive threshold)
- Univariate and multivariate
- Uses sklearn for nearest-neighbour computations

## Implementation

Adapted from the original MATLAB implementation by Yang Cao (Deakin University).

- Origin: adapted from MATLAB iCID
- Source: https://github.com/IsolationKernel/iCID
- Licence: GPLv3 (Copyright (c) 2023, Yang Cao, Deakin University)
- Licence file: `LICENSE` in this directory

## Citation

```bibtex
@article{cao2024icid,
  title   = {Isolation Distributional Kernel Change-Interval Detection},
  author  = {Cao, Yang and others},
  journal = {Journal of Artificial Intelligence Research},
  year    = {2024}
}
```
