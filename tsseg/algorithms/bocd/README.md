# BOCD (offline Bayesian change-point detection)

Offline Bayesian change-point inference using a conjugate Normal-Gamma prior.
Forward and backward messages compute separate marginal probabilities of
boundaries, which are then thresholded to emit change points. Despite the
historical class name, the detector uses the whole series (Fearnhead, 2006),
rather than the online filtering algorithm of Adams & MacKay (2007).

## Key properties

- Type: change point detection
- Unsupervised (threshold-based) or semi-supervised (`max_cps`)
- Univariate and multivariate (two strategies: L2 norm reduction or per-channel ensembling)
- Constant hazard with geometric segment durations and configurable conjugate prior
- O(n²) worst-case time and O(n) inference memory; O(d n²) time for ensembling

## Implementation

Originally adapted from the reference implementation by hildensia
(https://github.com/hildensia/bayesian_changepoint_detection). The core
inference routines are in `bayesian_models.py`. The detector now uses a
forward-backward recursion for boundary marginals, with stable running moments,
instead of constructing an ordinal change-point matrix. The generic legacy
helpers are retained for compatibility and are not called by the detector.

The final segment is right censored. Boundary marginals sum to the posterior
expected number of boundaries, not one. `truncate` is a natural-log cutoff
(`exp(truncate)`); all segment moments are evaluated even when a message sum is
truncated, so worst-case time remains quadratic. `max_cps` caps the thresholded
output; it does not condition inference on a known count.

The Gaussian model detects changes of mean or variance, not arbitrary temporal
dynamics. L2 reduction discards channel direction; per-channel ensembling avoids
the direction loss but does not model joint covariance changes.
Earlier benchmark results used a constant duration mass, unit-sum boundary
scores and cubic inference, and must be rerun for this corrected detector.

- Origin: adapted from hildensia/bayesian_changepoint_detection
- Licence: MIT (Copyright (c) 2014 Johannes Kulick)
- Licence file: `LICENSE` in this directory

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
