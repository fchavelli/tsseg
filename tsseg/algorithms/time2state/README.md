# Time2State

Unsupervised state detection using learned temporal representations. A Causal
CNN encoder maps sliding windows to an embedding space, then a Dirichlet
Process Gaussian Mixture Model (DPGMM) clusters the embeddings to produce state
labels.

## Key properties

- Type: state detection
- Unsupervised or semi-supervised (`n_states` is an upper bound for the DPGMM)
- Univariate and multivariate
- Requires PyTorch

## Implementation

Taken from the reference implementation by Chengyu Wang et al. The core
encoder and clustering logic is in `time2state.py`.

- Origin: taken from https://github.com/Lab-ANT/Time2State
- Licence: MIT (Copyright (c) 2022 Chengyu Wang)
- Licence file: `LICENSE` in this directory

## Citation

```bibtex
@article{wang2023time2state,
  title   = {{Time2State}: An Unsupervised Framework for Inferring the
             Latent States in Time Series Data},
  author  = {Wang, Chengyu and Wu, Kui and Zhou, Tongqing and Cai, Zhiping},
  journal = {Proceedings of the ACM on Management of Data},
  volume  = {1},
  number  = {1},
  pages   = {1--18},
  year    = {2023},
  doi     = {10.1145/3588697}
}
```
