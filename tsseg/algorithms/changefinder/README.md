# ChangeFinder

Online change-point detection via two-stage outlier score learning.
Reduces the change-point detection problem to outlier detection by
applying a pair of sequentially discounting autoregressive (SDAR)
models with moving-average smoothing.

## Key properties

- Type: change point detection
- Unsupervised or semi-supervised (`n_cps`)
- Univariate and multivariate (two strategies: L2 norm reduction or per-channel ensembling)
- Online, single-pass algorithm with linear time complexity
- Configurable scoring function and smoothing window

## Algorithm overview

ChangeFinder operates in two stages:

1. **Stage 1 — Outlier scoring.** An SDAR model (AR model with
   exponential discounting) is fitted incrementally on the raw time
   series. At each time step the model produces a one-step-ahead
   prediction; the discrepancy between prediction and observation
   yields an *outlier score* (logarithmic or quadratic loss). These
   scores are smoothed with a causal moving average of width *T*.

2. **Stage 2 — Change-point scoring.** A second, independent SDAR
   model is fitted on the smoothed outlier scores from Stage 1. The
   same scoring procedure produces a second score curve, which is
   again smoothed. Peaks in this *change-point score* correspond to
   locations where the data-generating process transitions between
   states.

Change points are selected via peak-picking on the final curve
(`scipy.signal.argrelextrema`), with optional thresholding and
minimum-distance constraints.

### SDAR (Sequential Discounting AR)

The core building block is an AR(*k*) model updated incrementally with
an exponential discount factor *r* ∈ (0, 1):

- **Initialisation:** Yule-Walker equations on a batch prefix.
- **Online update:** at each step, the AR coefficients, mean, auto-
  covariances, and residual variance are blended between the old
  estimates and the contribution of the new observation, weighted by
  *r*. The Toeplitz system is re-solved at each step.

A small *r* (e.g. 0.005) remembers longer history, whereas a larger
*r* (e.g. 0.05) forgets faster and is more sensitive to recent
changes.

## Parameters

| Parameter                | Type       | Default          | Description                                                                                      |
|--------------------------|------------|------------------|--------------------------------------------------------------------------------------------------|
| `order`                  | int        | `5`              | AR order *k* for both SDAR stages.                                                              |
| `discount`               | float      | `0.005`          | Discounting rate *r* ∈ (0, 1). Controls forgetting speed.                                       |
| `smooth_window`          | int        | `7`              | Moving-average window length *T* applied after each SDAR stage.                                  |
| `score`                  | str        | `"logarithmic"`  | Scoring function: `"logarithmic"` (neg. log-likelihood) or `"quadratic"` (squared error).        |
| `n_cps`                  | int / None | `None`           | Number of change points to return. If `None`, all peaks above threshold are returned.            |
| `threshold`              | float / None | `None`         | Minimum score for a peak. If `None`, uses `mean + 2*std` of the score curve.                     |
| `min_distance`           | int        | `10`             | Minimum number of samples between successive change points.                                      |
| `multivariate_strategy`  | str        | `"l2"`           | Strategy for multivariate data: `"l2"` (L2 norm reduction) or `"ensembling"` (per-channel).      |
| `tolerance`              | float      | `0`              | Tolerance for aggregating change points across channels (ensembling only).                       |
| `axis`                   | int        | `0`              | Time axis.                                                                                       |

## Implementation

The implementation consists of two modules:

- `sdar.py` — The `SDAR` class implementing the Sequential Discounting
  AR model with batch initialisation (Yule-Walker) and online update.
  Uses `scipy.linalg.solve` with a Toeplitz system and small
  regularisation to prevent singularity.

- `detector.py` — The `ChangeFinderDetector` class (inherits
  `BaseSegmenter`), orchestrating the two-stage pipeline, moving-
  average smoothing, peak selection, and multivariate strategies.

### Implementation details

- **Initialisation region:** The first `max(2k, 30)` observations are
  used to bootstrap each SDAR model via Yule-Walker. The init region
  scores are back-filled with the median of valid scores to prevent
  an artificial jump at the boundary from being detected as a change
  point.

- **Stage 2 warm-up:** The second SDAR model starts after
  `init_len + T` steps to ensure the smoothed scores have stabilised.

- **Multivariate support:** For `"l2"`, the multivariate series is
  reduced to a univariate signal via L2 norm before the pipeline. For
  `"ensembling"`, the pipeline runs independently per channel and
  change points are aggregated by temporal proximity.

## Usage

```python
from tsseg.algorithms import ChangeFinderDetector

# Basic usage with default parameters
detector = ChangeFinderDetector()
cps = detector.fit_predict(X)

# Customised: faster forgetting, quadratic scoring
detector = ChangeFinderDetector(
    order=3,
    discount=0.05,
    smooth_window=5,
    score="quadratic",
)
cps = detector.fit_predict(X)

# Semi-supervised: request exactly 5 change points
detector = ChangeFinderDetector(n_cps=5)
cps = detector.fit_predict(X)
```

## Citation

```bibtex
@ARTICLE{changefinder2006,
  author   = {Takeuchi, J. and Yamanishi, K.},
  journal  = {IEEE Transactions on Knowledge and Data Engineering},
  title    = {A unifying framework for detecting outliers and change
              points from time series},
  year     = {2006},
  volume   = {18},
  number   = {4},
  pages    = {482-492},
  doi      = {10.1109/TKDE.2006.1599387}
}
```
