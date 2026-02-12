# E2USD (Efficient-yet-Effective Unsupervised State Detection)

Deep learning method for unsupervised state detection. Learns temporal
representations with a Deformable Dense Encoding Module (DDEM), then clusters
them with a Dirichlet Process Gaussian Mixture Model (DPGMM) to produce state
labels.

## Key properties

- Type: state detection
- Unsupervised or semi-supervised (`n_states` is an upper bound)
- Univariate and multivariate
- Requires PyTorch

## Implementation

Taken from the original repository and wrapped with a `BaseSegmenter`-compatible
detector. The core logic is in `e2usd.py`.

- Origin: taken from https://github.com/AI4CTS/E2Usd
- Licence: none found in the original repository

## Citation

```bibtex
@inproceedings{chen2024e2usd,
  title     = {{E2Usd}: Efficient-yet-Effective Unsupervised State Detection
               for Multivariate Time Series},
  author    = {Chen, Zhichen and others},
  booktitle = {Proceedings of the ACM Web Conference (WWW)},
  year      = {2024}
}
```
