# TS-CP2 (Time Series Change Point Detection based on Contrastive Predictive Coding)

Change point detection via contrastive predictive coding. Learns temporal
representations by predicting future windows and detects change points by
measuring representation dissimilarity across sliding windows.

## Key properties

- Type: change point detection
- Unsupervised or semi-supervised
- Univariate and multivariate
- Requires TensorFlow and the `tcn` package
- Multiple contrastive loss functions: NCE, DCL, focal, hard-DCL
- Multiple similarity metrics: cosine, dot, euclidean, edit

## Implementation

TensorFlow reimplementation adapted from the original code by Shohreh Deldari
et al. The encoder is in `network.py` and loss functions in `losses.py`.

- Origin: adapted from original TS-CP2 code: https://github.com/cruiseresearchgroup/TSCP2
- Source: Deldari et al. (RMIT / University of Melbourne)
- Licence: not specified in the original work

## Citation

```bibtex
@inproceedings{deldari2021tscp2,
  title     = {{TS-CP2}: Change Point Detection in Multivariate Time Series
               via Contrastive Learning},
  author    = {Deldari, Shohreh and Smith, Daniel V. and Xue, Hao and Salim, Flora D.},
  booktitle = {Proceedings of the Web Conference (WWW)},
  pages     = {1930--1941},
  year      = {2021},
  doi       = {10.1145/3442381.3449855}
}
```
