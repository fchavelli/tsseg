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
- Each window gets one state, so `window_size` (256 by default) sets the time
  resolution and should be well below the length of the shortest segment;
  the authors' experiments use 128 to 512 points. On a 1000-point series of
  three segments of 300 to 400 points, only 5 of the 15 default windows lie
  inside a single segment (ARI 0.22 guided; 0.90 to 0.95 with
  `window_size=32, step=32`)

## Implementation

Taken from the reference implementation by Chengyu Wang et al. The core
encoder and clustering logic is in `time2state.py`.

- Origin: taken from https://github.com/Lab-ANT/Time2State
- Licence: MIT (Copyright (c) 2022 Chengyu Wang)
- Parts from other projects, in `time2state.py`:
  - the causal CNN encoder (`Chomp1d`, `SqueezeChannels`,
    `CausalConvolutionBlock`, `CausalCNN`, `CausalCNNEncoder`) and `Dataset`:
    taken from the code of Franceschi et al. (2019),
    https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries,
    on which Time2State builds; Apache License 2.0 (Copyright 2019
    Jean-Yves Franceschi, Aymeric Dieuleveut, Martin Jaggi);
  - `Chomp1d` and `CausalConvolutionBlock` are there partly taken from
    https://github.com/locuslab/TCN; MIT (Copyright (c) 2018 CMU Locus Lab).
- Licence file: `LICENSE` in this directory, with the three licences, the
  `NOTICE` of Franceschi et al. and the changes made to their code

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
