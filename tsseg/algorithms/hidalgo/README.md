# Hidalgo

Bayesian clustering algorithm based on the estimation of local intrinsic
dimensionality. Uses Gibbs sampling to assign data points to clusters with
different intrinsic dimensions.

## Key properties

- Type: state detection
- Semi-supervised
- Multivariate
- Uses nearest-neighbour distances for dimensionality estimation

## Implementation

Adapted from the aeon toolkit with a numerical stability fix: the `sample_p`
step in the Gibbs sampler has been moved to the log domain to prevent
overflow/underflow when cluster counts are large.

- Origin: adapted from aeon (with numerical stability fix)
- Licence: BSD 3-Clause (aeon toolkit)
