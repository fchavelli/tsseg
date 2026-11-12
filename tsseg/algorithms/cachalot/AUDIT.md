# Cachalot integration audit, 2026-11-12

The implementation follows the three stages in Section 3 of Harchaoui and
Lévy-Leduc, [*Catching Change-points with Lasso*, NIPS 2007](https://papers.neurips.cc/paper_files/paper/2007/file/e5841df2166dd424a57127423d276bbe-Paper.pdf).
Algebraic reference tests pass, and the optimized implementation retains all
36 single-channel predictions on the nine bundled MoCap trials. This does
**not** establish reproduction of the paper's Monte Carlo results. The
smaller seeded study below shows appreciable oversegmentation, particularly
with few true changes. Multivariate support is a channel union, not a joint
estimator of common change points.

## Paper-to-code mapping

| Paper | Implementation and checks |
| --- | --- |
| Eq. (4), squared error with total variation constraint | Unpenalized signal level, centered step columns at positions 1 through n-1, no column standardization. The lower-triangular Eq. (3) includes a constant first column; penalizing that column literally would penalize the initial level, contrary to Eq. (4). We implement the TV formulation. |
| Section 3, first Kmax Lasso variables | First entries along the centered, unweighted Lasso path. Compared against `sklearn.linear_model.lars_path(method="lasso")` on 40 seeded path/budget combinations, including complete paths. Exact-fit paths stop before entering zero-correlation noise. Tied entries use increasing position order and the budget may truncate a tie. |
| Eq. (10), reduced dynamic programming | Full-signal segment SSE minimized over every candidate subset size. Costs and reconstructed segmentations are checked against exhaustive enumeration. Prefix sums and cached candidate-interval costs avoid repeated SSE calculations. |
| Model selection, ratio J(k+1)/J(k) | First ratio at least 1-nu; includes k=0, consistent with Tables 1 and 2 despite the displayed k>=1 rule. If no ratio qualifies, use all available candidates. The published objectives select four changes at nu=0.05. |
| Section 5.1 settings | Defaults Kmax=50 and nu=0.01 retained. No tuning to force agreement with the publication. |
| Index convention | A step at paper variable beta_30 starts sample 30 (one-based): tsseg index 29. Two newly seeded realizations of the toy design test this convention. They are not the unpublished realization in Table 1. |

Centering is an intentional interpretation of Eq. (4), rather than a claim
that the two displayed objectives are literally identical. The 2010 JASA
paper is a related reference; this audit targets the 2007 algorithm and does
not claim to reproduce the journal experiments.

## Reproduced defects and corrections

- On four exact levels of 30 samples each, scaling the amplitudes by 1e-14
  made the old path return only split 90 instead of 30, 60, 90. Its absolute
  event tolerance incorrectly treated the next entry as the endpoint.
  Normalize the centered response and use a relative cumulative-sum error
  bound. This changes neither Lasso variable order nor the TV penalty.
- A constant array of 120 copies of 0.1 previously returned candidate 119.
  Subtract an actual represented observation before centering, so constant
  inputs are exactly zero rather than mean-rounding residuals.
- The interval parameter schema admits NaN. Cachalot now rejects a nonfinite
  `nu` explicitly instead of silently selecting the maximum count.

The first two failures were captured before the correction. New tests also
cover ties, very short series, candidate caps, invalid constructor values,
the published ratio table, and both axis conventions. Existing generic tests
were already passing; they had missed these numerical defects.

## Structured path and complexity

For centered steps at positions p, the Gram kernel is
`G(p,q) = min(p,q) - p*q/n`. On ordered active positions, its inverse is
tridiagonal. With signs s and zero signs at boundaries 0 and n,

```text
(inv(G_A) s)_i = (s_i - s_(i-1)) / (p_i - p_(i-1))
                + (s_i - s_(i+1)) / (p_(i+1) - p_i).
```

Each direction has its coefficient's sign, or is zero, so this unweighted
step design has no Lasso dropouts. `G_all,A inv(G_A) s` is linear
interpolation of the active signs, with zero endpoints. Parameterizing the
path by the decrease in residual correlation avoids equiangular
normalization and any dense linear solve. This is the same Lasso candidate
path, not a greedy split approximation.

For K candidates and d channels, time is `O(d (n K + K^3))`. Per-channel
working memory is `O(n + K^2)`; stored paths and reconstructed subsets add
`O(d K^2)`. The former code rebuilt n-by-active Gram blocks and solved an
active system on every step. The reduced DP retains cubic asymptotic time,
but precomputes SSEs and uses vectorized predecessor comparisons.

## Performance and multivariate behavior

Python 3.12.12, NumPy 1.26.4, OpenBLAS limited to one thread. Median of three
runs on seeded Gaussian input, measured separately for path and DP.
The machine is shared, so these are indicative local measurements.

| n | K | Original total (s) | Current total (s) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 50 | 0.0632 | 0.00574 | 11.0x |
| 10,000 | 50 | 0.428 | 0.0117 | 36.5x |
| 10,000 | 200 | 3.76 | 0.0888 | 42.3x |

All nine four-channel bundled MoCap trials, unchanged defaults, no
normalization, tsseg one-to-one F1 matching with margin 1% of each trial:

| Aggregate | Original | Current |
| --- | ---: | ---: |
| Macro union F1 | 0.403688 | 0.403688 |
| Macro union precision | 0.303556 | 0.303556 |
| Macro union recall | 0.655115 | 0.655115 |
| Mean single-channel F1 | 0.357190 | 0.357190 |
| Mean oracle best-channel F1 | 0.575767 | 0.575767 |
| Total processing time, 36 channels (s) | 11.45 | 0.443 |

Every channel's predicted indices are identical before and after, not just
the aggregate scores. The oracle channel is chosen using the reference and
is not a deployable selection rule.

The union preserves independent channel decisions. It pools neither weak
evidence nor noise estimates, merges no nearby detections, and may return up
to d*K points. Each channel may have different true ruptures. The above
precision/recall tradeoff is insufficient evidence for recommending it as a
joint multivariate method. For shared ruptures, a separate implementation
of [Bleakley and Vert's group fused Lasso](https://arxiv.org/abs/1106.4199)
would be the relevant method; it is not implemented by this PR.

## Statistical validation and outstanding gap

The script also generates 2,400 seeded signals from the Section 5.1 model:
n=500, K in {5,10,15,20}, Gaussian levels, minimum segment size 10, Gaussian
noise with standard deviation m times the minimum absolute jump, m in
{0.5,1.5}, Kmax=50 and nu=0.01. There are 30 independent layout/level pairs
per cell, each with 10 noise replicates, and a fixed seed 7301. Matching
uses tsseg F1 with an explicitly specified five-sample margin.

| K | m | Mean precision | Mean recall | Mean F1 | Mean predicted K |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.5 | 0.586 | 0.999 | 0.717 | 9.70 |
| 5 | 1.5 | 0.523 | 0.869 | 0.633 | 9.48 |
| 10 | 0.5 | 0.715 | 0.999 | 0.827 | 14.45 |
| 10 | 1.5 | 0.676 | 0.916 | 0.771 | 14.01 |
| 15 | 0.5 | 0.825 | 0.999 | 0.902 | 18.36 |
| 15 | 1.5 | 0.807 | 0.948 | 0.869 | 17.82 |
| 20 | 0.5 | 0.895 | 0.996 | 0.942 | 22.35 |
| 20 | 1.5 | 0.878 | 0.938 | 0.905 | 21.47 |

This is a smaller study, not a replication of Tables 3-5. The paper crosses
30 layouts with 30 level configurations and uses 10 noise realizations per
combination; it does not fully specify the sampling law for layouts, level
variance, or a numerical matching tolerance. It publishes no seeds. Missing
seeds prevent recovering the original realizations, but do not preclude
statistical replication. In particular, precision 0.586 at K=5, m=0.5 is
below Table 3's reported 0.80, while recall is close to one. This gap remains
unresolved: neither algorithmic reference tests nor high recall certify the
published statistical performance. The original PR's claim of smoke tests
on 75 TSSB series is not treated as verified evidence in this audit.

No exact reproduction test for the well-log figure is claimed: the paper
does not specify the median-filter window or a numeric reference output.

## Re-run and artifacts

From the repository root, in an environment with the test dependencies:

```sh
NUMBA_CACHE_DIR=/tmp/tsseg-numba-cache python -m pytest tests/algorithms \
    -k 'cachalot or CachalotDetector' -q -p no:cacheprovider
OPENBLAS_NUM_THREADS=1 python -m tests.reproduction.cachalot_benchmark \
    --baseline-ref 8633a8c --output /tmp/cachalot-audit.json
```

The baseline commit must be available locally. The benchmark downloads
nothing. Its JSON contains timings, runtime metadata, all per-trial/channel
predictions and scores, and the synthetic aggregates. The audited run is
stored at [tests/reproduction/cachalot_audit_2026-11-12.json](../../../tests/reproduction/cachalot_audit_2026-11-12.json).

Local validation: 98 targeted/generic Cachalot tests passed, 6 skipped;
263 schema/dependency/package integration tests passed, 24 skipped.
The original PR's completed Python 3.11 and 3.14 jobs had passed; the other
matrix jobs and documentation were cancelled, not reported as test failures.
Full CI validation of the updated branch must be checked separately.
