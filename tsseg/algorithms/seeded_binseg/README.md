# SeedBS (Seeded Binary Segmentation) and WBS (Wild Binary Segmentation)

Binary segmentation computed on a fixed collection of intervals instead of on
the segments of the current solution: the best split of every interval is a
candidate, and a selection rule keeps some of them. Seeded intervals (layers of
evenly spread, overlapping intervals whose length decays geometrically) make
the search near-linear and deterministic; random intervals give wild binary
segmentation.

## Key properties

- Type: change point detection
- Unsupervised (strengthened Schwarz information criterion, or a threshold
  `penalty` on the gain) or semi-supervised (`n_cps`)
- Selection: greedy (largest gain first) or narrowest over threshold
- Gain: decrease of a vendored ruptures cost (`l2`: the squared CUSUM statistic)
- O(n log n) cost evaluations for the seeded intervals
- Univariate and multivariate

## Implementation

Written from the papers only. The authors' code
([kovacssolt/SeedBinSeg](https://github.com/kovacssolt/SeedBinSeg)) has no
license and the R packages `wbs`, `not` and `breakfast` are GPL-2.0-only: none
was opened.

- `_core.py`: seeded intervals (Kovács et al., 2023, Definition 1), random
  intervals (Fryzlewicz, 2014, Sec. 3.3), greedy selection (Fryzlewicz, 2014),
  narrowest-over-threshold selection and its threshold path (Baranowski et al.,
  2019; the path is followed by recomputing a solution only from the first
  newly eligible interval that it accepts), sSIC (Fryzlewicz, 2014, Sec. 3.4,
  eq. (4)).
- WBS (`intervals="wild"`, `selection="greedy"`) applies the optional
  augmentation of Fryzlewicz (2014, Sec. 3.3) at the top level only: the whole
  series is a candidate, the segments between selected change points are not
  added at later steps.
- `detector.py`: the best split and gain of each interval come from the
  vendored ruptures `Binseg.single_bkp` (`jump=1`), on the numba backend when
  it is available.
- Paper version used: arXiv:2002.06633v1 (the only arXiv version); section,
  theorem and table numbers in the code follow it, and those of Baranowski et
  al. (2019) follow its arXiv version.
- Extension not in the papers: for a multivariate series the sSIC gives each
  channel its own variance and counts d + 1 parameters per change point (the
  general form of Baranowski et al., 2019, eq. (3.1)). The gain sums the
  channels unweighted: channels with different noise levels must be put on the
  same noise level first.

## Source

- Kovács, S., Bühlmann, P., Li, H., Munk, A. (2023). Seeded binary segmentation:
  a general methodology for fast and optimal changepoint detection.
  Biometrika, 110(1), 249-256.
- License of this implementation: AGPL-3.0-only, as tsseg.

## BibTeX

```bibtex
@article{kovacs2023seeded,
  title   = {Seeded binary segmentation: a general methodology for fast and
             optimal changepoint detection},
  author  = {Kov{\'a}cs, Solt and B{\"u}hlmann, Peter and Li, Housen and
             Munk, Axel},
  journal = {Biometrika},
  volume  = {110},
  number  = {1},
  pages   = {249--256},
  year    = {2023},
  doi     = {10.1093/biomet/asac052}
}

@article{fryzlewicz2014wild,
  title   = {Wild binary segmentation for multiple change-point detection},
  author  = {Fryzlewicz, Piotr},
  journal = {The Annals of Statistics},
  volume  = {42},
  number  = {6},
  pages   = {2243--2281},
  year    = {2014},
  doi     = {10.1214/14-AOS1245}
}

@article{baranowski2019narrowest,
  title   = {Narrowest-over-threshold detection of multiple change points and
             change-point-like features},
  author  = {Baranowski, Rafal and Chen, Yining and Fryzlewicz, Piotr},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {81},
  number  = {3},
  pages   = {649--672},
  year    = {2019},
  doi     = {10.1111/rssb.12322}
}
```
