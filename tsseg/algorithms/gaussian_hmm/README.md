# Gaussian HMM

Hidden Markov model with diagonal Gaussian emissions, fitted by Baum-Welch (EM)
and decoded by Viterbi. The number of states is given (`n_states`) or chosen by
BIC (`n_states=None`).

## Key properties

- Type: state detection
- Unsupervised (BIC over `1..max_states`) or semi-supervised (`n_states`)
- Univariate and multivariate
- One EM iteration costs O(nK²); numba speeds up the recursions
- No duration model: suits regimes that differ in level or spread

## Implementation

- Origin: new code, written for tsseg from Rabiner (1989)
- `_core.py`: log-space forward-backward (posterior marginals and expected
  transition counts) and Viterbi, compiled by numba when available
- `detector.py`: k-means initialisation, M-step, BIC selection
- Checked against hmmlearn (BSD-3): same parameters and log-likelihood after
  each EM iteration from the same start (`tests/algorithms/test_gaussian_hmm.py`,
  skipped when hmmlearn is not installed), and against exact enumeration on a
  small model
- Licence: AGPL-3.0, as tsseg

## Citation

```bibtex
@article{rabiner1989tutorial,
  title   = {A Tutorial on Hidden {Markov} Models and Selected Applications in Speech Recognition},
  author  = {Rabiner, Lawrence R.},
  journal = {Proceedings of the IEEE},
  volume  = {77},
  number  = {2},
  pages   = {257--286},
  year    = {1989}
}
```
