# HMM (Hidden Markov Model)

Simple univariate HMM with Gaussian emissions. Parameters are estimated via the
Baum-Welch (EM) algorithm, and the most likely state sequence is decoded with
the Viterbi algorithm. State boundaries define the change points.

## Key properties

- Type: state detection
- Semi-supervised (requires number of states)
- Univariate only
- Pure Python (no external HMM library required)

## Implementation

Adapted from the aeon toolkit.

- Origin: adapted from aeon
- Source: https://github.com/aeon-toolkit/aeon/blob/v1.3.0/aeon/segmentation/_hmm.py
- Licence: BSD 3-Clause (aeon toolkit)

## Citation

```bibtex
@article{rabiner1989tutorial,
  title   = {A Tutorial on Hidden {M}arkov Models and Selected Applications
             in Speech Recognition},
  author  = {Rabiner, Lawrence R.},
  journal = {Proceedings of the IEEE},
  volume  = {77},
  number  = {2},
  pages   = {257--286},
  year    = {1989}
}
```
