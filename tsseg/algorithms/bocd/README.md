# BOCD (Bayesian Online Change-Point Detection)

Offline Bayesian change-point inference using a conjugate Normal-Gamma prior.
Evaluates every possible segment via dynamic programming and computes posterior
probabilities of boundaries, which are then thresholded to emit change points.

## Key properties

- Type: change point detection
- Unsupervised (threshold-based) or semi-supervised (`max_cps`)
- Univariate and multivariate (two strategies: L2 norm reduction or per-channel ensembling)
- Configurable hazard function and prior hyperparameters

## Implementation

Adapted from the reference implementation by hildensia
(https://github.com/hildensia/bayesian_changepoint_detection). The core
inference routine is in `bayesian_models.py`.

- Origin: adapted from hildensia/bayesian_changepoint_detection
- Licence: Apache License 2.0

## Citation

```bibtex
@article{fearnhead2006exact,
  title   = {Exact and Efficient Bayesian Inference for Multiple Changepoint Problems},
  author  = {Fearnhead, Paul},
  journal = {Statistics and Computing},
  volume  = {16},
  pages   = {203--213},
  year    = {2006}
}

@article{adams2007bayesian,
  title   = {Bayesian Online Changepoint Detection},
  author  = {Adams, Ryan P. and MacKay, David J. C.},
  journal = {arXiv preprint arXiv:0710.3742},
  year    = {2007}
}
```
