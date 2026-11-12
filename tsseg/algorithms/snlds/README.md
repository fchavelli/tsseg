# SNLDS (Switching Non-Linear Dynamical System)

Learns discrete regimes through a switching nonlinear latent dynamical system.
Discrete states are marginalized exactly during training; observation encoding
is bidirectional. The detector remains experimental: improved diagnostics do
not establish reliable accuracy across datasets.

## Current implementation

- Guided state detection: `n_states` is supplied; reference labels are never
  used for fitting or initialization.
- The fast default posterior is factorized conditional on bidirectional
  observations. The paper's autoregressive posterior is available through
  `inference_type="autoregressive"`.
- Learned positive Gaussian noise, tracked initial priors and stateless seeds.
- Reconstruction initialization and latent k-means use observations only;
  `init_steps=0` disables them. Position-based initialization can be unsuitable
  for dynamical states distinguished by velocity rather than position.
- Conditional Viterbi at continuous posterior means is the default. This is a
  plug-in approximation, not integration over continuous uncertainty.
- Raw observations by default. `feature_window > 1` and `downsample > 1` expose
  a separate local-moment profile for activity statistics. These are explicit
  adaptations, not preprocessing from the paper.
- Requires TensorFlow, TensorFlow Probability and tf-keras (`tsseg[snlds]`).

```python
from tsseg.algorithms import SNLDSDetector

raw = SNLDSDetector(n_states=4, random_state=0).fit_predict(X)
activity = SNLDSDetector(
    n_states=4, feature_window=128, downsample=6, random_state=0
).fit_predict(X)
```

The documentation records all seeds of the fixed validation, including
failures. Profiles are not selected using the reference labels of each case.
Explicit configurations in external benchmarks override constructor defaults
and need separate reevaluation.

## Source and modifications

The authors' Apache-2.0 TensorFlow code is vendored from Google Research:
https://github.com/google-research/google-research/tree/5b09c22d73a9d35eb6c5d2a99b95677a45053466/snlds
(`forward_backward_algo.py`, `model_base.py`, `model_cavi_snlds.py`, `utils.py`,
`examples/config_utils.py`). Its license is `vendor/LICENSE`.

Modified files identify the changes: relative imports, Keras 3 recurrent
states, destination/source indexing, exact collapsed likelihood, tracked
initial priors, dynamic sequence lengths and stateless noise. `components.py`,
`decoding.py` and `detector.py` are tsseg's implementation of positive Gaussian
parameters, factorized inference, Viterbi, initialization and the estimator.

The exact likelihood without the sensitive pointwise cross-entropy term
follows Appendix B.4.1 of [Ansari et al. (2021)](https://arxiv.org/html/2110.13878v1).
No RED-SDS source is copied. There is no explicit duration model. Initialization,
factorization, decoding, local features and new budgets are adaptations,
not a reproduction of the original experimental protocol.

## Diagnostics

See `examples/diagnostics/README.md`, `snlds_audit.py`, `snlds_validate.py` and
`snlds_report.py`. Historical measurements in `snlds_audit_results.json`
predate the current model; the new fixed validation has a separate raw record.

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
