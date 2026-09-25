# HDP-HSMM (Hierarchical Dirichlet Process Hidden Semi-Markov Model)

Bayesian non-parametric state detection using Gibbs sampling: the weak-limit HDP-HSMM of
Johnson & Willsky (2013), with Gaussian emissions (Normal-Inverse-Wishart prior) and
shifted Poisson durations. The number of states is inferred up to `n_max_states`.

## Key properties

- Type: state detection
- Fully unsupervised (non-parametric; infers number of states)
- Univariate and multivariate
- NumPy/SciPy; numba (`tsseg[accelerators]`) compiles the forward messages

## Implementation

`detector.py` follows Johnson & Willsky (2013) and was checked against the authors'
[pyhsmm](https://github.com/mattjj/pyhsmm) and against exact enumeration. Its deviations
from the paper are listed in the module docstring.

- Origin: new code (replaces earlier pyhsmm-based implementation)

## Citation

Cite Johnson & Willsky (2013); the model was introduced in Johnson & Willsky (2010).

```bibtex
@article{johnson2013bayesian,
  title   = {Bayesian Nonparametric Hidden Semi-{M}arkov Models},
  author  = {Johnson, Matthew J. and Willsky, Alan S.},
  journal = {Journal of Machine Learning Research},
  volume  = {14},
  pages   = {673--701},
  year    = {2013}
}

@inproceedings{johnson2010hdphsmm,
  title     = {The Hierarchical {D}irichlet Process Hidden Semi-{M}arkov Model},
  author    = {Johnson, Matthew J. and Willsky, Alan S.},
  booktitle = {Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2010}
}
```
