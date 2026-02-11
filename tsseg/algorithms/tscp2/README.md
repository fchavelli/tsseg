# TS-CP2 (Time Series Change Point Detection with Self-Supervised Contrastive Predictive Coding)

TensorFlow implementation of TS-CP2. The model learns representations via
contrastive predictive coding and detects change points by measuring
representation dissimilarity across sliding windows.

Requires `tensorflow`.

## References

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
