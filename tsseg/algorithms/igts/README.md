# IGTS (Information Gain Temporal Segmentation)

Top-down segmentation algorithm that recursively splits the time series at the
point maximising information gain, treating each candidate segment as a
histogram of discretised values.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised (step size controls granularity)
- Multivariate only

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
