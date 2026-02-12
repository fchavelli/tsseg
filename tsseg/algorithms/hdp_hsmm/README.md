# HDP-HSMM (Hierarchical Dirichlet Process Hidden Semi-Markov Model)

Bayesian non-parametric state detection using Gibbs sampling. Each state's
emission is modelled with a Normal-Inverse-Wishart prior and state durations
follow a Negative-Binomial distribution. The number of states is inferred
automatically from the data.

## Key properties

- Type: state detection
- Fully unsupervised (non-parametric; infers number of states)
- Univariate and multivariate
- Pure Python (no external HMM library required)

## Implementation

New pure-Python implementation of the HDP-HSMM Gibbs sampler, replacing an
earlier dependency on the `pyhsmm` package. A legacy detector wrapping `pyhsmm`
is kept in `legacy_detector.py` / `legacy_pyhsmm.py` for reference.

- Origin: new code (replaces earlier pyhsmm-based implementation)

## Citation

```bibtex
@article{johnson2013bayesian,
  title   = {Bayesian Nonparametric Hidden Semi-{M}arkov Models},
  author  = {Johnson, Matthew J. and Willsky, Alan S.},
  journal = {Journal of Machine Learning Research},
  volume  = {14},
  pages   = {673--701},
  year    = {2013}
}
```
