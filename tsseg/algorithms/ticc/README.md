# TICC (Toeplitz Inverse Covariance-based Clustering)

Segments multivariate time series by jointly learning Toeplitz-structured
inverse covariance matrices for each cluster and assigning windows to clusters
via an EM-like procedure with a temporal consistency penalty.

## Key properties

- Type: state detection
- Unsupervised (`n_states=None`: number of states chosen by BIC, as the paper
  recommends) or semi-supervised (`n_states` given)
- Multivariate (models temporal cross-covariance structure)
- Transductive: `predict` returns the labels computed during `fit`

## Implementation

Taken from the original repository by David Hallac (Stanford). The core solver
is in `ticc.py`; its changes from the original, which leave the output
unchanged, are listed at the top of the file (local random generator, solver
run in-process, vectorised Z-update and assignment step). The BIC used to select
the number of states is described in the docstring of `TiccDetector`; the
original repository's `computeBIC`, kept in `ticc.py`, sums the log-likelihoods
of the clusters without weighting them by their size and selects one state on
every series tried.

- Origin: taken from https://github.com/davidhallac/TICC
- Licence: BSD 2-Clause (Copyright (c) 2017-2018, David Hallac, Sagar Vare, Saachi Jain, and Others)
- Licence file: `LICENSE` in this directory

## Citation

```bibtex
@inproceedings{hallac2017toeplitz,
  title     = {Toeplitz Inverse Covariance-Based Clustering of
               Multivariate Time Series Data},
  author    = {Hallac, David and Vare, Sagar and Boyd, Stephen and Leskovec, Jure},
  booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference
               on Knowledge Discovery and Data Mining},
  pages     = {215--223},
  year      = {2017},
  doi       = {10.1145/3097983.3098060}
}
```
