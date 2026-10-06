# SNLDS (Switching Non-Linear Dynamical System)

Learns discrete regimes with a switching non-linear dynamical system trained by
collapsed amortized variational inference (CAVI-SNLDS). Each regime has its own
MLP transition on a continuous latent state; the discrete states are integrated
out with the forward-backward algorithm, and each time step gets the regime of
highest posterior probability.

## Key properties

- **Experimental**: below a random segmentation on a 54-series sample of TSB-SEG
  (labels change about 200 times too often); kept for further study

- Type: state detection
- Semi-supervised (`n_states` is the number of regimes)
- Univariate and multivariate
- Requires TensorFlow, TensorFlow Probability and `tf-keras` (`pip install tsseg[snlds]`)
- Slow: about 0.1 s per gradient step on one CPU thread with the defaults

## Implementation

`detector.py` is a tsseg wrapper. The model code in `vendor/` is the authors'
TensorFlow code.

- Origin: https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/snlds
  (`forward_backward_algo.py`, `model_base.py`, `model_cavi_snlds.py`, `utils.py`,
  `examples/config_utils.py`)
- Changes: relative imports; Keras 3 compatible initial state and call of the
  posterior RNN in `model_cavi_snlds.py`. Each modified file says so in its header.
- Licence: Apache License 2.0 (Copyright The Google Research Authors)
- Licence file: `vendor/LICENSE`

Training follows the annealing schedule of the paper's Reacher experiment,
scaled to `n_train_steps`, on random windows of the series (see the class
docstring).

## Citation

```bibtex
@inproceedings{dong2020collapsed,
  title     = {Collapsed Amortized Variational Inference for Switching Nonlinear Dynamical Systems},
  author    = {Dong, Zhe and Seybold, Bryan and Murphy, Kevin and Bui, Hung},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {2638--2647},
  year      = {2020}
}
```
