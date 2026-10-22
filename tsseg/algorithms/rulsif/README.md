# RuLSIF

Change-point detection by relative density-ratio estimation (Liu, Yamada,
Collier and Sugiyama, 2013). Each split is scored by the symmetrised
α-relative Pearson divergence between the subsequences before and after it,
estimated by RuLSIF; change points are peaks of the score.

## Key properties

- Type: change point detection
- Unsupervised (peaks above `threshold`) or semi-supervised (`n_cps`)
- Univariate and multivariate (subsequences concatenate the channels)
- Cost: a cross-validated kernel fit per split and direction, about 8 ms per
  point with the defaults

## Implementation

- Origin: port of the authors' MATLAB code, `change_detection.m`,
  `RelULSIF.m`, `demo.m` and `lib/`
  (https://github.com/anewgithubname/change_detection, commit `496042e`,
  repository of the first author, S. Liu), MIT licence; its notice is kept in
  `LICENSE` in this directory.
- Permission: on 2026-10-09 the tsseg maintainer reported that M. Yamada
  (co-author) authorised consulting the MATLAB code (his words, verbatim:
  "makoto m'a autorisé", i.e. "Makoto authorised me"). The MIT licence of the
  repository above covers the port in any case.
- History: a first version was written from the paper alone; it was then
  aligned on the MATLAB code (per-window standardisation, base kernel width,
  random folds) and now matches a line-by-line NumPy transcription of that
  code to rounding.
- `rulsif.py`: subsequences, cross-validation folds, the RuLSIF estimator of
  the α-relative Pearson divergence (closed form, cross-validation over
  kernel widths and regularisation parameters), and the score of every split
- `detector.py`: `RuLSIFDetector`, peak selection
- Defaults are the paper's (k = 10, n = 50, α = 0.1, five kernel-width factors and
  five regularisation parameters, 5-fold cross-validation, Section 4). Where
  the code is more specific than the paper (standardisation, median
  heuristic, folds), the code is followed; the class docstring lists these
  points, the score alignment and the unsupervised threshold (tsseg
  choices).
- Licence: AGPL-3.0 for tsseg, MIT for the ported code (`LICENSE`)

## Citation

```bibtex
@article{liu2013change,
  title   = {Change-point detection in time-series data by relative density-ratio estimation},
  author  = {Liu, Song and Yamada, Makoto and Collier, Nigel and Sugiyama, Masashi},
  journal = {Neural Networks},
  volume  = {43},
  pages   = {72--83},
  year    = {2013},
  doi     = {10.1016/j.neunet.2013.01.012}
}
```
