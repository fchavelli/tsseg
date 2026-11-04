# SST

Singular spectrum transformation: change-point detection by comparing the
dominant subspaces of a past and a future Hankel matrix (local linear
dynamics), with the fast randomised SVD of Weber & Lenz (2025).

## Key properties

- Type: change point detection
- Semi-supervised only (`n_cps`): the papers give a score, not a rule to pick
  change points without their number
- Univariate and multivariate (Hankel matrices of the channels side by side,
  one shared subspace)
- Cost: `O(q (k + p) d N log N)` per time point with the randomised SVD
  (default), `O(d N^3)` with the full SVD; about 5 s for 20,000 points with
  the defaults

## Implementation

- Origin: reimplemented from the papers, without third-party code: geometry,
  score, lag and preprocessing from Idé & Tsuda (2007, Sections 2 and 5.1),
  preprocessing also in Idé & Inoue (2005, Section 4.2); randomised SVD with
  FFT Hankel products from Weber & Lenz (2025, Algorithms 1, 3-5, defaults of
  Section IV-E)
- `sst.py`: preprocessing, Hankel windows, exact and FFT rSVD left singular
  vectors, score
- `detector.py`: `SSTDetector`, peak selection
- tsseg choices (no paper fixes them): `n_cps` highest peaks at least
  `min_distance` apart (default `window_size`); no unsupervised mode;
  multivariate Hankel matrices concatenated column-wise, as multivariate
  SSA stacks Page matrices (Alanqary, Alomar & Shah, NeurIPS 2021, Sec. 2,
  Eq. 4, Hankel variant in footnote 1); score 0 where the
  matrices do not fit (Idé & Tsuda use periodic boundaries); one Gaussian test
  matrix per series; QR at every subspace iteration (Weber & Lenz use LU in
  the intermediate ones, for speed)
- Licence: AGPL-3.0 (tsseg)

## Citation

```bibtex
@inproceedings{ide2007change,
  title     = {Change-Point Detection using {K}rylov Subspace Learning},
  author    = {Id{\'e}, Tsuyoshi and Tsuda, Koji},
  booktitle = {Proceedings of the 2007 SIAM International Conference on Data Mining},
  pages     = {515--520},
  year      = {2007},
  doi       = {10.1137/1.9781611972771.54}
}

@article{weber2025accelerating,
  title   = {Accelerating Singular Spectrum Transformation for Scalable Change Point Detection},
  author  = {Weber, Lucas and Lenz, Richard},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {213556--213577},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3640386}
}

@inproceedings{ide2005knowledge,
  title     = {Knowledge Discovery from Heterogeneous Dynamic Systems using Change-Point Correlations},
  author    = {Id{\'e}, Tsuyoshi and Inoue, Keisuke},
  booktitle = {Proceedings of the 2005 SIAM International Conference on Data Mining},
  pages     = {571--575},
  year      = {2005},
  doi       = {10.1137/1.9781611972757.63}
}
```
