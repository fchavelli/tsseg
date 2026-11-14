# BOCD (offline Bayesian change-point detection)

Gaussian product-partition inference with a geometric duration prior. The
historical class name is retained; the detector uses the whole series,
following Fearnhead (2006), and joint MAP decoding follows Fearnhead and Liu
(2007). It does not run the causal filter of Adams and MacKay (2007).

## Defaults

- Native multivariate Gaussian likelihood with a full covariance matrix.
- Normal-inverse-Wishart prior: `nu0 = 2*alpha+d-1`, `Psi0 = 2*beta*I`.
  In one dimension this is exactly Normal-Gamma.
- Standardise each channel; `mu=0`, `kappa=alpha=1`, `beta=10`, prior run
  length `hazard_lambda=1000` samples before conditioning on grid and length
  restrictions, minimum segment length 25.
- Joint MAP partition, optionally restricted to at most `max_cps` boundaries.
- At most 2,000 interior candidate locations. All observations enter the
  likelihood; no block-mean downsampling. `max_candidates=None` uses all positions.

The priors were calibrated on 15 series from TSSB, SKAB, HAS, MoCap and UTSA,
with synthetic checks that rejected an overly conservative variance prior.
These defaults still assume independent Gaussian observations within each regime.
They need not describe every oscillation or dynamical change in sensor signals.

## Alternative modes

`decoding="marginal"` thresholds local peaks of boundary probabilities (default
threshold 0.5). Their sum is the expected change count, not one. `truncate=None`
uses all terms; finite legacy truncation may omit important terms and does not
reduce likelihood work. Neither parameter affects MAP decoding.

`multivariate_strategy="l2"` reduces to a norm before standardising the resulting
scalar series. It loses direction. `"ensembling"` runs per channel, aggregates,
and spaces the resulting boundaries; it does not model joint covariance.
`standardize=False` interprets priors in the original measurement units.

Inference is conditional on the candidate grid and the minimum segment length.
The duration prior retains sample-time units, so changing the grid changes the
conditioned model. With `m` candidate vertices, native inference costs
O(n d² + m² d³) time and O(n + m d²) working memory; a binding count constraint K
adds O(m² K) time and O(m K) memory. Univariate inference costs O(n + m²) time.
Stable running moments and Cholesky evaluation avoid segment matrices; an
eigenvalue fallback preserves a tiny prior ridge at deficient rank.

## Origin and validation

Originally adapted from [hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection).
The adapted generic helpers remain in `bayesian_models.py` for compatibility.
The detector uses `_offline.py`; its Gaussian evidence is checked against
sequential multivariate Student-t predictions, and MAP partitions and marginal
probabilities against exhaustive enumeration. A pinned upstream comparison is
part of the local defaults study; no new runtime dependency is introduced.

Earlier benchmark results used a constant duration mass, unit-sum boundary
scores and cubic inference. The new likelihood, decoder, grid and defaults
change detection outputs; those results must be recomputed.

Explicit values in existing experiment configurations override the new defaults.
Update or remove old `beta` and `hazard_lambda` overrides; probability-threshold
and truncation grid axes are inactive with MAP decoding.

- Origin of legacy helpers: hildensia/bayesian_changepoint_detection
- Licence of adapted code: MIT (Copyright (c) 2014 Johannes Kulick)
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

@article{fearnhead2007online,
  title   = {On-line Inference for Multiple Changepoint Problems},
  author  = {Fearnhead, Paul and Liu, Zhen},
  journal = {Journal of the Royal Statistical Society: Series B},
  volume  = {69},
  number  = {4},
  pages   = {589--605},
  year    = {2007}
}

@article{adams2007bayesian,
  title   = {Bayesian Online Changepoint Detection},
  author  = {Adams, Ryan P. and MacKay, David J. C.},
  journal = {arXiv preprint arXiv:0710.3742},
  year    = {2007}
}
```
