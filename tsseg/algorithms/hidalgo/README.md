# Hidalgo - Heterogeneous Intrinsic Dimension Algorithm

Bayesian state detector that segments multivariate data into regions of
different **local intrinsic dimensionality** (ID).  Each data point is
assigned to one of $K$ manifolds, where each manifold $k$ has its own
estimated dimension $d_k$.

> Allegra, M. et al. "Data segmentation based on the local intrinsic
> dimension." *Scientific Reports* 10.1 (2020): 1-12.
> <https://www.nature.com/articles/s41598-020-72222-0>

---

## Method overview

### Core idea - TWO-NN extended to mixtures

For each data point $x_i$, let $r_{i,1}$ and $r_{i,2}$ be the distances to
its first and second nearest neighbour.  The ratio

$$\mu_i = \frac{r_{i,2}}{r_{i,1}}$$

follows a Pareto distribution $f(\mu \mid d) = d \cdot \mu^{-(d+1)}$ when
the density is locally approximately constant (TWO-NN assumption).

Hidalgo extends this to a mixture of $K$ Pareto components: the data are
assumed to live on the union of $K$ manifolds with dimensions
$d_1, \dots, d_K$.  A latent variable $z_i \in \{1, \dots, K\}$ indicates
which manifold point $i$ belongs to.

### Bayesian model

The posterior is:

$$P(\mathbf{z}, \mathbf{d}, \mathbf{p} \mid \boldsymbol{\mu}, \mathcal{N}^{(q)}) \;\propto\; P(\boldsymbol{\mu} \mid \mathbf{z}, \mathbf{d}) \;\cdot\; P(\mathcal{N}^{(q)} \mid \mathbf{z}) \;\cdot\; P(\mathbf{z} \mid \mathbf{p}) \;\cdot\; P(\mathbf{d}) \;\cdot\; P(\mathbf{p})$$

where:

- $P(\boldsymbol{\mu} \mid \mathbf{z}, \mathbf{d})$ - Pareto likelihood
  given manifold assignment
- $P(\mathcal{N}^{(q)} \mid \mathbf{z})$ - Potts-like local homogeneity
  term encouraging neighbours to share the same label, controlled by
  $\zeta$
- $P(\mathbf{z} \mid \mathbf{p})$ - categorical prior with mixing
  probabilities $\mathbf{p}$
- $P(\mathbf{d})$ - $\text{Gamma}(a, b)$ priors on the dimensions
  (default: $a = b = 1$)
- $P(\mathbf{p})$ - $\text{Dir}(c)$ prior on mixing probabilities
  (default: $c = 1$)

The posterior is sampled via **Gibbs sampling**.  Multiple random restarts
(`n_replicas`) are run and the chain with the highest average
log-posterior is kept.  The final assignment uses the marginal posterior
$\pi_{ik} = P(z_i = k)$ averaged over post-burn-in samples:

- If $\max_k \pi_{ik} \ge 0.8$ then $z_i = \arg\max_k \pi_{ik}$
- Otherwise $z_i = -1$ (uncertain assignment)

### This is a state detector, not a change-point detector

Hidalgo assigns a **label** to every data point based on the manifold it
belongs to.  Output is a dense label vector ($0, 1, \dots, K{-}1$, or
$-1$ for uncertain points).  It does not detect transitions; it classifies
points.

### What $K$ represents

$K$ is the **number of distinct manifolds** (i.e. number of states), not a
dimension.  Each manifold $k$ has its own estimated intrinsic dimension
$d_k$.  Two spatially separated regions with the same dimension are treated
as distinct manifolds.

---

## Key properties

| Property               | Value                            |
|------------------------|----------------------------------|
| Detector type          | state detection                  |
| Supervision            | semi-supervised ($K$ must be set)|
| Univariate support     | **no** (see below)               |
| Multivariate support   | yes (requires dim >= 2)          |
| Returns                | dense label vector               |

---

## Parameters

| Parameter        | Default       | Description                                       |
|------------------|--------------|---------------------------------------------------|
| `metric`         | `"euclidean"` | Distance metric (passed to sklearn NearestNeighbors)|
| `K_states`       | 1            | Number of manifolds / states                       |
| `zeta`           | 0.8          | Local homogeneity level $\zeta \in (0, 1)$         |
| `q`              | 3            | Local homogeneity range (number of neighbours)     |
| `n_iter`         | 1000         | Number of Gibbs sampling iterations                |
| `n_replicas`     | 1            | Number of random restarts                          |
| `burn_in`        | 0.9          | Fraction of iterations discarded as burn-in        |
| `sampling_rate`  | 10           | Keep every *sampling_rate*-th post-burn-in sample  |
| `fixed_Z`        | False        | If True, do not sample Z (estimate d, p only)      |
| `use_Potts`      | True         | Enable Potts-like local homogeneity term           |
| `estimate_zeta`  | False        | Update $\zeta$ during sampling                     |
| `seed`           | 0            | Random seed for reproducibility                    |

The paper recommends `q=3, zeta=0.8` as the optimal working point.

---

## Example

```python
import numpy as np
from tsseg.algorithms.hidalgo import HidalgoDetector

rng = np.random.default_rng(42)

# 100 points in 5D: first 60 on a 2D surface, last 40 in full 5D
X = np.zeros((100, 5))
X[:60, :2] = rng.standard_normal((60, 2))      # low-dimensional manifold
X[60:, :] = rng.standard_normal((40, 5))        # high-dimensional manifold

model = HidalgoDetector(K_states=2, n_iter=500, burn_in=0.8, seed=42)
labels = model.fit_predict(X, axis=0)

# labels[i] in {0, 1, -1}
# model._d  -> estimated dimensions per manifold, e.g. [~2, ~5]
# model._p  -> estimated mixing proportions
```

---

## Why univariate time series are not supported

Hidalgo discriminates regions by contrasting their **intrinsic
dimensionality**.  For a univariate signal the data points live on a 1-D
line, so the intrinsic dimension is uniformly $d = 1$ everywhere: there
is no heterogeneity to detect.

### Workaround: time-delay embedding

A univariate series $x_t$ can be lifted to $\mathbb{R}^m$ via Takens
embedding:

$$X_t = (x_t,\; x_{t+\tau},\; x_{t+2\tau},\; \dots,\; x_{t+(m-1)\tau})$$

In embedding space, different dynamical regimes may occupy manifolds of
different dimensions (e.g. a periodic orbit is a 1-D loop vs. a chaotic
attractor with fractional-D).  Hidalgo can then segment these regimes.

This preprocessing is **not built into the detector** because it introduces
unrelated hyperparameters (embedding dimension $m$, delay $\tau$) and is
a general technique applicable to any multivariate algorithm.

### Experimental result: time-delay embedding does not work on CPD benchmarks

An experimental wrapper (`univariate/embedding_detector.py`) was
implemented and tested on TSSB (75 series) and UTSA (32 series) with
$m \in \{3, 5, 10\}$, $\tau \in \{1, 2\}$, and semi-supervised $K$.
**Mean ARI ≈ 0.009** (random-level) across all configurations.

Oracle TWO-NN analysis confirmed the root cause: TSSB/UTSA segments differ
by signal *morphology* (z-normalised UCR patterns), not by *intrinsic
dimensionality*. In embedding space, different waveform shapes still occupy
manifolds of similar dimension, so Hidalgo has no discriminating signal.

The only positive result was on a synthetic (sine vs white noise) signal
where true ID differs ($d \approx 1$ vs $d \approx 10$): ARI = 0.57.

Full analysis and results are documented in `univariate/README.md`.

---

## Implementation notes

Adapted from the [aeon toolkit](https://github.com/aeon-toolkit/aeon)
implementation (`aeon.segmentation._hidalgo`).

### Changes from the original aeon code

1. **`sample_p` log-domain fix**: the acceptance ratio in the Dirichlet
   sampler for $\mathbf{p}$ is computed in the log domain to prevent
   overflow / underflow when cluster counts are large.

2. **Zero-distance stability** (aeon commit `b66706b`): $\mu_i$ computation
   guards against zero-distance neighbours (duplicated data points) by
   returning $\mu = 1$ instead of dividing by zero.

3. **Empty-sampling guard** (aeon commit `dbe19a0`): when burn-in and
   sampling-rate filtering discard all Gibbs samples, a `ValueError` with
   actionable guidance is raised instead of crashing with an `IndexError`.
   The posterior-probability normalisation uses the actual sample count
   (`bestsampling.shape[0]`) rather than the `idx` length.

### Licence

BSD 3-Clause (aeon toolkit)
