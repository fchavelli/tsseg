# ClaSP and CLaP

This directory contains two related algorithms that share core routines
(scoring, segmentation, distance, validation, nearest-neighbour search, window
size selection). Both are exposed through `__init__.py`.

---

## ClaSP (Classification Score Profile)

Parameter-free time series segmentation. A binary classifier is trained to
distinguish the left and right halves of the series at each candidate split
point; the resulting classification score profile is analysed for peaks that
indicate change points.

### Key properties

- Type: change point detection
- Unsupervised or semi-supervised (`n_change_points`)
- Univariate and multivariate
- Detector class: `ClaspDetector` (in `clasp_detector.py`)

---

## CLaP (Classification Label Profile)

Extension of ClaSP for state detection. After identifying change points with
ClaSP, time series classifiers from the `aeon` library are used to assign a
state label to each segment.

### Key properties

- Type: state detection
- Unsupervised or semi-supervised
- Univariate and multivariate
- Detector class: `ClapDetector` (in `clap_detector.py`)

---

## File layout

```
clap/
  clasp_detector.py     ClaspDetector (change point detection)
  clap_detector.py      ClapDetector  (state detection)
  clap.py               CLaP core logic
  segmentation.py       BinaryClaSPSegmentation (shared recursive segmenter)
  scoring.py            Classification score profile computation
  distance.py           Distance functions
  nearest_neighbour.py  kNN utilities
  validation.py         Statistical validation of split points
  window_size.py        Automatic window size selection (SUSS)
  utils.py              Shared helpers
```

## Implementation

Adapted from the original ClaSP implementation by Arik Ermshaus.

- Origin: adapted from aeon / original ClaSP code by Arik Ermshaus
- Licence: BSD 3-Clause (Copyright (c) 2023, Arik Ermshaus)
- Licence file: `LICENSE` in this directory

## Citation

```bibtex
@article{ermshaus2023clasp,
  title   = {{ClaSP}: Parameter-Free Time Series Segmentation},
  author  = {Ermshaus, Arik and Sch{\"a}fer, Patrick and Leser, Ulf},
  journal = {Data Mining and Knowledge Discovery},
  volume  = {37},
  pages   = {1262--1300},
  year    = {2023}
}
```
