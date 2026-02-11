# IGTS (Information Gain Temporal Segmentation)

Implementation of the IGTS algorithm for activity recognition. IGTS
iteratively places change points at positions that maximise the information
gain (reduction in entropy) of the resulting segmentation, using a top-down
splitting strategy.

## Source

Adapted from `aeon` implementation: https://github.com/aeon-toolkit/aeon/blob/v0.5.0/aeon/annotation/igts.py

Licence: BSD 3-Clause License

## References

```bibtex
@article{sadri2017information,
  title   = {Information Gain-based Metric for Recognizing Transitions
             in Human Activities},
  author  = {Sadri, Amin and Ren, Yongli and Salim, Flora D.},
  journal = {Pervasive and Mobile Computing},
  volume  = {38},
  pages   = {92--109},
  year    = {2017},
  doi     = {10.1016/j.pmcj.2017.01.003}
}
```
