# TICC (Toeplitz Inverse Covariance-based Clustering)

Segments multivariate time series by jointly learning Toeplitz-structured
inverse covariance matrices for each cluster and assigning windows to clusters
via an EM-like procedure with a temporal consistency penalty.

## Key properties

- Type: state detection
- Semi-supervised (requires `n_states`)
- Multivariate (models temporal cross-covariance structure)
- Transductive: `predict` returns the labels computed during `fit`

## Implementation

Taken from the original repository by David Hallac (Stanford). The core solver
is in `ticc.py`.

- Origin: taken from https://github.com/davidhallac/TICC
- Licence: not specified in the original repository

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
