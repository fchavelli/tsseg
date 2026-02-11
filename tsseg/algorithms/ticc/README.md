# TICC (Toeplitz Inverse Covariance-based Clustering)

Wrapper around a local implementation of TICC. The algorithm segments
multivariate time series by jointly learning Toeplitz-structured inverse
covariance matrices for each cluster and assigning windows to clusters
via an EM-like procedure with a temporal consistency penalty.

Note: TICC is a transductive (fit-and-predict) algorithm — `_predict`
returns the labels computed during `_fit`.

## References

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
