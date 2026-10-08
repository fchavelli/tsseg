# RuLSIF

Change-point detection by relative density-ratio estimation (Liu, Yamada,
Collier and Sugiyama, 2013). Each split is scored by the symmetrised
α-relative Pearson divergence between the subsequences before and after it,
estimated by RuLSIF; change points are peaks of the score.

## Key properties

- Type: change point detection
- Unsupervised (peaks above `threshold`) or semi-supervised (`n_cps`)
- Univariate and multivariate (subsequences concatenate the channels)
- Cost: a cross-validated kernel fit per split and direction, about 5 ms per
  point with the defaults

## Implementation

- Origin: new code, written for tsseg from the paper only. The authors'
  MATLAB code has no licence and was not consulted.
- `rulsif.py`: subsequences, the RuLSIF estimator of the α-relative Pearson
  divergence (closed form, cross-validation over kernel widths and
  regularisation parameters), and the score of every split
- `detector.py`: `RuLSIFDetector`, peak selection
- Defaults are the paper's (k = 10, n = 50, α = 0.1, five kernel widths and
  five regularisation parameters, 5-fold cross-validation, Section 4). Choices
  the paper leaves open (score alignment, median distance, fold assignment,
  unsupervised threshold) are listed in the class docstring.
- Licence: AGPL-3.0, as tsseg

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
