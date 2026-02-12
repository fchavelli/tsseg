# Time2State

Unsupervised state detection using learned temporal representations. A Causal
CNN encoder maps sliding windows to an embedding space, then a Dirichlet
Process Gaussian Mixture Model (DPGMM) clusters the embeddings to produce state
labels.

## Key properties

- Type: state detection
- Unsupervised (`n_states` is an upper bound for DPGMM)
- Univariate and multivariate
- Requires PyTorch

## Implementation

Taken from the original repository by Kunpeng Zheng et al. The core encoder and
clustering logic is in `time2state.py`.

- Origin: taken from https://github.com/Lab-ANT/Time2State
- Licence: not specified in the original repository

## Citation

```bibtex
@article{zheng2023time2state,
  title   = {{Time2State}: An Unsupervised Framework for Inferring the
             Latent State in Time Series Data},
  author  = {Zheng, Kunpeng and others},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {17},
  number  = {1},
  year    = {2023}
}
```
