# Metrics

This module provides evaluation metrics for time series segmentation. All metrics inherit from `BaseMetric` and expose a common `compute(y_true, y_pred)` interface returning a dictionary with at least a `score` key.

## Change-Point Detection Metrics

### F1 Score

Standard margin-based F1 for change-point detection. A predicted change point is considered a true positive if it falls within a fixed tolerance window of a ground-truth change point.

### Hausdorff Distance

Worst-case deviation between the set of true and predicted change points.

### Gaussian F1 Score

A soft alternative to the margin-based F1. Instead of a binary tolerance window, predictions are rewarded according to a Gaussian kernel centred on each true change point:

$$
G(p) = \exp\!\left( -\frac{(p - c)^2}{2\sigma^2} \right), \qquad \sigma = \alpha \cdot T
$$

where $T$ is the series length and $\alpha$ (default 0.01) controls the width. A greedy bipartite matching maximises the total Gaussian reward $W$, from which soft precision ($W / |P|$) and recall ($W / |C|$) are computed. The Gaussian F1 is their harmonic mean.

**Advantages over classic F1:**
- Eliminates the arbitrary tolerance hyperparameter.
- Smooth partial credit instead of binary accept/reject.

```python
from tsseg.metrics import GaussianF1Score

metric = GaussianF1Score()
result = metric.compute(y_true=[0, 250, 500], y_pred=[0, 260, 500])
# {'score': 0.96, 'precision': 0.96, 'recall': 0.96, 'matched_weight': 0.96}
```

### Bidirectional Covering

Extends the classical covering metric by measuring overlap in both directions:

- **Ground-truth coverage** $C_\text{GT}$: how well each true segment is covered by predictions.
- **Prediction coverage** $C_\text{Pred}$: how well each predicted segment is covered by the ground truth.

For segment sets $\mathcal{S}$ and $\mathcal{T}$, the directional covering is:

$$
\mathrm{Cover}(\mathcal{S} \to \mathcal{T}) = \frac{\sum_{s \in \mathcal{S}} |s| \max_{t \in \mathcal{T}} \mathrm{IoU}(s, t)}{\sum_{s \in \mathcal{S}} |s|}
$$

The two directions are aggregated with a configurable strategy: `harmonic` (default), `geometric`, `arithmetic`, or `min`.

```python
from tsseg.metrics import BidirectionalCovering

metric = BidirectionalCovering()
result = metric.compute(y_true=[0, 50, 120, 200], y_pred=[0, 60, 180, 200])
# {'score': 0.84, 'ground_truth_covering': 0.87, 'prediction_covering': 0.82}
```

### Covering

Classical (unidirectional) covering score measuring ground-truth coverage only.

## State Detection Metrics

### Adapted from scikit-learn

The following metrics are standard clustering measures adapted from [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) for the time series segmentation setting:

| Metric | Description |
|---|---|
| `AdjustedRandIndex` | [Adjusted Rand Index](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html) — Rand index adjusted for chance agreement |
| `AdjustedMutualInformation` | [Adjusted Mutual Information](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html) — mutual information adjusted for chance |
| `NormalizedMutualInformation` | [Normalised Mutual Information](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html) — mutual information normalised by joint entropy |

### Introduced in *Toward Interpretable Evaluation Measures for Time Series Segmentation*

The following metrics were introduced in [Toward Interpretable Evaluation Measures for Time Series Segmentation](https://arxiv.org/abs/2510.23261) (NeurIPS 2025). They address key shortcomings of existing point-based measures by accounting for the temporal structure of segmentation errors.

| Metric | Description |
|---|---|
| `WeightedAdjustedRandIndex` | **WARI** — a temporal-aware extension of ARI that weights errors by their position within segments, giving more importance to errors that disrupt the temporal structure of the segmentation |
| `WeightedNormalizedMutualInformation` | **WNMI** — a temporal-aware extension of NMI using the same positional weighting scheme as WARI |
| `StateMatchingScore` | **SMS** — a fine-grained, interpretable metric that uses Hungarian matching to align predicted and true states, then identifies and scores four distinct types of segmentation errors (over-segmentation, under-segmentation, misclassification, and boundary shift), allowing error-specific weighting |
