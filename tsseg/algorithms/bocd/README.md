# Bayesian Change-Point Detection (BOCD)

This detector implements the offline Bayesian change-point inference scheme
described by Fearnhead [1]. Given a univariate time series, the algorithm
evaluates every possible segment using a conjugate Normal-Gamma prior and
computes, via dynamic programming, the posterior probability that a boundary
occurs after each sample. Change points are selected by thresholding these
posterior probabilities and enforcing a minimum distance between accepted
boundaries.

## High-level workflow

1. **Preprocessing** -- the input series is aligned along the time axis (axis 0).
   Multivariate inputs are handled via one of two strategies (see below).
   The detector works on dense `numpy.ndarray` inputs and returns the indices
   where a boundary is detected.

2. **Prior specification** -- the user can configure the Normal-Gamma
   hyper-parameters `(mu, kappa, alpha, beta)` and the expected run length
   `hazard_lambda`. A constant hazard of `1 / hazard_lambda` is used to express
   the prior probability of encountering a change point.

3. **Dynamic programme** -- the core routine is adapted from
   `offline_changepoint_detection` in the
   [hildensia/bayesian_changepoint_detection][repo] project. For every pair of
   start/end indices it computes the marginal likelihood (integrating out mean
   and variance) and recursively accumulates log-probabilities of change-point
   configurations while truncating negligible contributions.

4. **Posterior aggregation** -- the log-probabilities of all configurations that
   place a boundary after sample `t` are summed in log-space, normalised, and
   converted to posterior probabilities. A simple greedy selector keeps the
   highest-scoring change points above `cp_prob_threshold`, obeying a
   `min_distance` constraint and the optional `max_cps` limit.

## Multivariate strategies

The underlying BOCD algorithm is univariate. Two strategies are available for
multivariate inputs (parameter `multivariate_strategy`):

| Strategy            | Description |
|---------------------|-------------|
| `"l2"` (default)    | Reduce to univariate via row-wise L2 norm across channels, then run BOCD once. Fast, but may mask change points when segment norms are similar. |
| `"ensembling"`      | Run BOCD independently on each channel and aggregate the detected change points via voting / clustering. More robust but slower (linear in `n_channels`). |

## Relationship to the reference implementation

The code is a distilled version of the original
[Bayesian Changepoint Detection][repo] repository, keeping only the
offline algorithm and removing auxiliary scripts, Cython extensions and the
online detector used in Adams & MacKay [2]. The `BOCDDetector` class wraps the
offline routine with an `aeon`-compatible API and exposes the most relevant
hyper-parameters to users.

## File layout

| File | Description |
|------|-------------|
| `detector.py` | `BOCDDetector` -- main segmenter class |
| `bayesian_models.py` | `offline_changepoint_detection` DP + online variant (unused) |
| `old/` | Deprecated files from the original hildensia repo (hazard functions, priors, data generation) |

## License

The reference implementation ([hildensia/bayesian_changepoint_detection][repo])
is released under the **Apache License 2.0**.

## References

1. Paul Fearnhead. *Exact and Efficient Bayesian Inference for Multiple
   Changepoint Problems*. Statistics and Computing, 16(2):203-213, 2006.
2. Ryan P. Adams and David J.C. MacKay. *Bayesian Online Changepoint
   Detection*. arXiv:0710.3742, 2007.
3. Xuan Xiang and Kevin Murphy. *Modeling Changing Dependency Structure in
   Multivariate Time Series*. ICML, pages 1055-1062, 2007.

```bibtex
@article{fearnhead2006exact,
  title   = {Exact and Efficient {B}ayesian Inference for Multiple Changepoint Problems},
  author  = {Fearnhead, Paul},
  journal = {Statistics and Computing},
  volume  = {16},
  number  = {2},
  pages   = {203--213},
  year    = {2006},
  doi     = {10.1007/s11222-006-8450-8}
}
```

[repo]: https://github.com/hildensia/bayesian_changepoint_detection
