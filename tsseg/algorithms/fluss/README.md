# FLUSS (Fast Low-cost Unipotent Semantic Segmentation)

Matrix-profile-based semantic segmentation. Computes the Matrix Profile, derives
an Arc Curve (corrected for regime-change bias), and detects segmentation
boundaries at the valleys of this curve.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised
- Univariate and multivariate
- Requires the `stumpy` library

## Implementation

Wrapper around the `stumpy` library's FLUSS implementation. The detector
handles multivariate input by applying FLUSS per channel and aggregating.

- Origin: wrapper around stumpy
- Licence: BSD 3-Clause (stumpy, https://github.com/TDAmeritrade/stumpy)

## Citation

```bibtex
@inproceedings{gharghabi2017matrix,
  title     = {Matrix Profile {VIII}: Domain Agnostic Online Semantic
               Segmentation at Superhuman Performance Levels},
  author    = {Gharghabi, Shaghayegh and Ding, Yifei and Yeh, Chin-Chia Michael
               and Kamgar, Kaveh and Ulanova, Liudmila and Keogh, Eamonn},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  pages     = {117--126},
  year      = {2017}
}
```
