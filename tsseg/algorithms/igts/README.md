# IGTS (Information Gain Temporal Segmentation)

Top-down segmentation algorithm that recursively splits the time series at the
point maximising information gain, treating each candidate segment as a
histogram of discretised values.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised (step size controls granularity)
- Univariate and multivariate

### Univariate support

Shannon entropy over a single channel is constant, so raw univariate input
cannot produce meaningful information gain.  Following Eq. 12-13 of the
original paper, univariate series are automatically augmented before the
search: each channel is normalised and its complement (`max − x`) is appended,
doubling the number of channels.  This breaks the constant-entropy degeneracy
and lets the algorithm detect change points in univariate data.

## Implementation

Adapted from the aeon toolkit.

- Origin: adapted from aeon
- Licence: BSD 3-Clause (aeon toolkit)

## Citation

```bibtex
@article{sadri2017information,
  title   = {Information Gain-based Metric for Recognizing Transitions in
             Human Activities},
  author  = {Sadri, Amin and Ren, Yongli and Salim, Flora D.},
  journal = {Pervasive and Mobile Computing},
  volume  = {38},
  pages   = {92--109},
  year    = {2017}
}
```
