# TICC (Toeplitz Inverse Covariance-based Clustering)

Segments multivariate time series by jointly learning Toeplitz-structured
inverse covariance matrices for each cluster and assigning windows to clusters
via an EM-like procedure with a temporal consistency penalty.

## Key properties

- Type: state detection
- Unsupervised (`n_states=None`: number of states chosen by BIC, as the paper
  recommends) or semi-supervised (`n_states` given)
- Multivariate (models temporal cross-covariance structure)
- `predict` returns the labels computed during `fit` on the series given to
  `fit`, and labels another series with a new fit, with a warning

## Implementation

Taken from the original repository by David Hallac (Stanford). The core solver
is in `ticc.py`. The port corrects the reference solver's truncated training
windows: every row contains `X[start:start + window_size]`, including the final
observations, and all complete windows contribute assignment costs and
likelihoods. The reference instead zero-padded the final windows and omitted
their costs. Fitted outputs, including BIC-selected state counts, can change
when `window_size > 1`; earlier TICC scores need to be recomputed. The labels
remain aligned to window starts, with the last label extended to the final
`window_size - 1` points; the unused `threshold` argument is retained.

The other changes listed at the top of `ticc.py` preserve successful outputs
(local random generator, solver run in-process, vectorised Z-update and
assignment step). The BIC used to select
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
