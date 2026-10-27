# E2USD (Efficient-yet-Effective Unsupervised State Detection)

Deep learning method for unsupervised state detection. Learns temporal
representations with a Decomposed Dual-view Embedding Module (DDEM), then clusters
them with a Dirichlet Process Gaussian Mixture Model (DPGMM) to produce state
labels.

## Key properties

- Type: state detection
- Unsupervised or semi-supervised (`n_states` is an upper bound)
- Univariate and multivariate
- Requires PyTorch
- Each window gets one state, so `window_size` (256 by default) sets the time
  resolution and should be well below the length of the shortest segment;
  the authors' experiments use 128 to 512 points. On a 1000-point series of
  three segments of 300 to 400 points, only 5 of the 15 default windows lie
  inside a single segment (ARI 0.28 to 0.52 guided over five seeds; 0.83 to
  0.95 with `window_size=32, step=32`)

## Implementation

Taken from the original repository and wrapped with a `BaseSegmenter`-compatible
detector. The core logic is in `e2usd.py`.

- Origin: taken from https://github.com/AI4CTS/E2Usd
- Licence: none found in the original repository
- Parts from other projects, in `e2usd.py`:
  - `Dataset`: taken, as in E2Usd, from the code of Franceschi et al. (2019),
    https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries;
    Apache License 2.0 (Copyright 2019 Jean-Yves Franceschi, Aymeric
    Dieuleveut, Martin Jaggi);
  - `moving_avg` and `series_decomp`: the decomposition blocks of Autoformer,
    https://github.com/thuml/Autoformer; MIT (Copyright (c) 2021 THUML @
    Tsinghua University).
- Licence file: `LICENSE` in this directory, for these parts only, with the
  `NOTICE` of Franceschi et al. and the changes made to the copied code

## Citation

```bibtex
@inproceedings{lai2024e2usd,
  title     = {{E2Usd}: Efficient-yet-effective Unsupervised State Detection
               for Multivariate Time Series},
  author    = {Lai, Zhichen and Li, Huan and Zhang, Dalin and Zhao, Yan and
               Qian, Weizhu and Jensen, Christian S.},
  booktitle = {Proceedings of the ACM Web Conference 2024},
  pages     = {3010--3021},
  year      = {2024},
  doi       = {10.1145/3589334.3645593}
}
```
