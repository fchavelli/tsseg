# HDP-HSMM (Hierarchical Dirichlet Process Hidden Semi-Markov Model)

Pure-Python implementation of the HDP-HSMM using Gibbs sampling.
Each state's emission is modelled with a Normal-Inverse-Wishart prior, and
state durations follow a Negative-Binomial distribution. The model
non-parametrically infers the number of states from data.

This implementation replaces the earlier dependency on the
[pyhsmm](https://github.com/mattjj/pyhsmm) package.

## References

```bibtex
@inproceedings{johnson2013bayesian,
  title     = {Bayesian Nonparametric Hidden Semi-{M}arkov Models},
  author    = {Johnson, Matthew J. and Willsky, Alan S.},
  booktitle = {Journal of Machine Learning Research},
  volume    = {14},
  pages     = {673--701},
  year      = {2013}
}
```
