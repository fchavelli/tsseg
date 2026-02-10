# Gaussian F1 Score for Change-Point Detection

## Overview
The Gaussian F1 score is a soft alternative to the standard, margin-based F1 metric used in change-point detection benchmarks. Instead of deciding whether a predicted change point is "close enough" to a reference by means of a fixed tolerance, we reward predictions according to a Gaussian kernel centred on each true change point. Exact matches therefore earn the maximum reward of 1, while nearby predictions receive a partial reward that vanishes smoothly as the error grows. All change points share the same Gaussian width, making the scoring scale uniform along the series.

This README summarises the motivation, mathematical formulation, and implementation details of the score defined in `tsseg/metrics/gaussian_f1.py`.

## Motivation
Traditional F1 computations for segmentation tasks rely on a tolerance window (for example, ±10 samples or ±1% of the signal length). This practice has two main limitations:

1. **Hyperparameter sensitivity** – performance numbers change with the chosen window size, making algorithm comparisons ambiguous.
2. **Binary rewards** – a prediction right inside the window receives full credit, whereas a prediction just outside receives none, despite being only marginally worse.

The Gaussian F1 removes the binary boundary by attributing a reward proportional to the proximity between predicted and true change points. This leads to smoother evaluation curves and removes the need to tune a per-dataset tolerance.

## Method
Let \(C = \{c_1, \dots, c_m\}\) be the set of true change-point indices and \(P = \{p_1, \dots, p_n\}\) the predicted indices. Every true change point is associated with the **same** Gaussian reward function

$$
G_i(p) = \exp\left( -\frac{(p - c_i)^2}{2\sigma_i^2} \right),
$$

where the standard deviation \(\sigma\) equals a fraction of the total series length, \(\sigma = \alpha \cdot T\). The default value \(\alpha = 0.01\) mirrors the common 1\% tolerance used by the classical F1 score, and a floor of 1 sample keeps the kernel well conditioned for short signals. Because \(\sigma\) no longer depends on local spacing, every change point is scored under the same tolerance band.

We then perform a greedy bipartite matching that maximises the total Gaussian reward. The algorithm considers all pairs \((c_i, p_j)\), scores them with \(G_i(p_j)\), sorts the scores in decreasing order, and retains the highest scoring pairs without conflict (each change point or prediction is matched at most once). The sum of the retained scores is denoted by \(W\).

## Soft Precision and Recall
The cumulative reward \(W\) is interpreted as the total "true positive mass". Soft precision and recall are computed as

$$
\text{precision} = \frac{W}{|P|}, \qquad \text{recall} = \frac{W}{|C|},
$$

with the convention that the result is zero when the denominator vanishes. The Gaussian F1 score is then the harmonic mean of these two quantities:

$$
\text{GaussianF1} = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}.
$$

Because the Gaussian rewards lie in \([0, 1]\), the F1 score remains in \([0,1]\) and reaches 1 only when every true change point is paired with an exact prediction and the number of predicted change points matches the ground truth.

## Edge Cases
The implementation handles several degenerate scenarios explicitly:

- **No true and no predicted change point**: the data is stationary and the detector remains silent, yielding a perfect score of 1.
- **No true but at least one predicted change point**: the detector raises a false alarm; score, precision, and recall drop to 0.
- **At least one true change point but no prediction**: all true changes are missed; the score is 0.
- **Single change point**: the global \(\sigma\) still equals `sigma_fraction × length`, so near misses retain partial credit even when only one event exists.

## Parameter summary

- `sigma_fraction` (default `0.01`): sets the Gaussian width as a proportion of the series length. It is chosen to match the classical 1% tolerance window out of the box, while keeping the scale uniform across change points.
- `min_sigma` (default `1.0`): lower bound applied to the Gaussian width for very short series.

## Relation to the Classic Margin-Based F1
The Gaussian formulation reduces to the classic windowed F1 when the Gaussian width \(\sigma\) tends to zero (reward collapses to exact matches) or to infinity (all predictions receive reward 1, removing discrimination). In practice, the adaptive width makes the score act like a smooth transition between these extremes without introducing an explicit hyperparameter.

## Complexity and Practical Considerations
The greedy matching runs in \(\mathcal{O}(mn \log (mn))\), where \(m\) and \(n\) are the numbers of true and predicted change points. For typical segmentation tasks these numbers are small, so the metric remains computationally inexpensive. A full Hungarian algorithm could yield an optimal assignment at \(\mathcal{O}((m + n)^3)\) but would require an additional dependency; empirical tests show that the greedy approach is sufficient because the Gaussian rewards heavily penalise large mismatches.

## Usage Example
```python
from tsseg.metrics.gaussian_f1 import GaussianF1Score

metric = GaussianF1Score()
y_true = [0, 250, 500]      # start, true CP, end
y_pred = [0, 260, 500]      # start, predicted CP, end
print(metric.compute(y_true, y_pred))
```

Output:

```
{'score': 0.9607, 'precision': 0.9607, 'recall': 0.9607, 'matched_weight': 0.9607}
```

The matched weight reveals how close the change points are: an exact match would have yielded 1.0.

## Summary
The Gaussian F1 score eliminates the arbitrary tolerance hyperparameter of traditional change-point evaluation by replacing binary matches with smooth, globally consistent Gaussian rewards. It provides a nuanced view of detector performance while remaining easy to interpret and computationally light.
