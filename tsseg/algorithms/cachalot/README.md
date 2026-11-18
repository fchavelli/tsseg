# Cachalot

Cachalot (CAtching CHAnge-points with LassO) detects changes in the mean of a
piecewise-constant signal (Harchaoui and Lévy-Leduc, NIPS 2007). It selects
candidates along a total-variation Lasso path, refines them with reduced
dynamic programming, and chooses the number of changes from the relative
decrease in residual sum of squares. The total-variation Lasso supplies the
candidates; Cachalot is the complete three-stage procedure named in the paper.

## Key properties

- Type: change point detection
- Fully unsupervised (`nu` controls model selection)
- Univariate method; multivariate input is processed independently by channel
- Per-channel cost: `O(n K + K^3)` time and `O(n + K^2)` auxiliary memory,
  where `n` is the series length and `K = min(max_candidates, n - 1)` for
  `n >= 2`

## Implementation

- Origin: independent reimplementation of Section 3 of the 2007 paper, with
  no third-party code reused.
- `_core.py`: Lasso-path candidates, least-squares dynamic programming over
  candidate subsets, and model selection. Centered step columns leave the
  signal level unpenalized, following the total-variation formulation in
  Eq. (4); columns are not standardized.
- The structured Lasso path uses `O(n K)` time and `O(n + K)` memory.
  Prefix sums give constant-time segment costs; reduced dynamic programming
  over all candidate-supported counts uses `O(n + K^3)` time and
  `O(n + K^2)` memory. These bounds describe the tsseg implementation.
- `detector.py`: `CachalotDetector`, with defaults `max_candidates=50` and
  `nu=0.01` from Section 5.1. It selects the first `k >= 0` such that
  `J(k + 1) / J(k) >= 1 - nu`, where `J(k)` is the minimum residual sum of
  squares with `k` candidate-supported changes. A zero residual also stops
  selection; if the threshold is never reached, all candidates are retained.
- Multivariate extension in tsseg: channels are processed sequentially and
  their selected change points are combined in a sorted union. Nearby
  detections are kept separately. For `d` channels, processing costs
  `O(d (n K + K^3))`, with an additional `O(d K log(d K))` sorting bound for
  the union. Total auxiliary memory is `O(n + d K^2)`, including retained
  dynamic-programming tables and per-channel results, excluding the input
  array.
- License: AGPL-3.0-only, as tsseg.

## Source

- Harchaoui, Z. and Lévy-Leduc, C. (2007).
  [*Catching Change-points with Lasso*](https://papers.neurips.cc/paper_files/paper/2007/file/e5841df2166dd424a57127423d276bbe-Paper.pdf).
  Advances in Neural Information Processing Systems 20. Algorithm source.
- Harchaoui, Z. and Lévy-Leduc, C. (2010).
  [*Multiple Change-Point Estimation With a Total Variation Penalty*](https://doi.org/10.1198/jasa.2010.tm09181).
  Journal of the American Statistical Association, 105(492), 1480-1493.
  Related journal article.

## Citation

```bibtex
@inproceedings{harchaoui2007catching,
  title     = {Catching Change-points with Lasso},
  author    = {Harchaoui, Zaid and L{\'e}vy-Leduc, C{\'e}line},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {20},
  year      = {2007},
  url       = {https://papers.neurips.cc/paper_files/paper/2007/hash/e5841df2166dd424a57127423d276bbe-Abstract.html}
}

@article{harchaoui2010multiple,
  title   = {Multiple Change-Point Estimation With a Total Variation Penalty},
  author  = {Harchaoui, Zaid and L{\'e}vy-Leduc, C{\'e}line},
  journal = {Journal of the American Statistical Association},
  volume  = {105},
  number  = {492},
  pages   = {1480--1493},
  year    = {2010},
  doi     = {10.1198/jasa.2010.tm09181}
}
```
