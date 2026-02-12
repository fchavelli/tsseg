# TIRE (TIme-invariant REpresentation)

Autoencoder-based change point detection that learns a partially
time-invariant representation tailored for CPD. The user can choose whether
change points should be sought in the time domain, frequency domain or both.
Detectable changes include abrupt shifts in mean, slope, variance,
autocorrelation and frequency spectrum.

## Key properties

- Type: change point detection
- Semi-supervised (`n_segments`) or unsupervised (prominence-based peak selection)
- Univariate and multivariate
- Requires PyTorch
- Domains: TD (time), FD (frequency) or both

## Implementation

PyTorch reimplementation of the original TensorFlow/Keras code by Tim De Ryck,
Maarten De Vos and Alexander Bertrand (KU Leuven / ETH Zurich). The detector
is wrapped in a `BaseSegmenter`-compatible class. Helper functions are in
`utils.py`.

- Origin: reimplemented from original code: https://github.com/deryckt/TIRE
- Licence: no licence in the original repository; authors granted use

## Citation

```bibtex
@article{deryck2021change,
  title   = {Change Point Detection in Time Series Data using Autoencoders
             with a Time-Invariant Representation},
  author  = {De Ryck, Tim and De Vos, Maarten and Bertrand, Alexander},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2021}
}
```
