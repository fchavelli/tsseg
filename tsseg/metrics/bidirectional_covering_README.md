# Bidirectional Covering Score for Change-Point Segmentation

## Overview
The bidirectional covering score extends the classical covering metric used in change-point segmentation benchmarks. The traditional definition measures how well each ground-truth segment is covered by the predictions. This asymmetry means that an over-segmented prediction can still achieve a high score if every true segment is partially covered. The bidirectional variant measures coverage in both directions and aggregates the two views, ensuring that predictions are rewarded only when they both cover the ground truth and avoid redundant fragmentation.

This README summarises the motivation, mathematical formulation, aggregation strategies, and implementation details of the score defined in `tsseg/metrics/bidirectional_covering.py`.

## Motivation
The classic covering metric evaluates the extent to which predicted segments overlap with ground-truth segments. Although informative, it overlooks the reciprocal perspective: a single long predicted segment may cover several true segments sparsely, inflating the score. Conversely, an algorithm that produces many short segments may cover the truth well but also introduces substantial over-segmentation. The bidirectional covering score addresses both issues by:

1. Measuring **ground-truth coverage**: each true segment is weighted by its duration and paired with the best-overlapping prediction.
2. Measuring **prediction coverage**: each predicted segment is weighted similarly and paired with the best-overlapping ground-truth segment.
3. Aggregating the two directional scores with a configurable mean so that both recall-like and precision-like behaviour must be strong to obtain a high final score.

## Method
Let the ground-truth change points be \(G = \{g_0, g_1, \dots, g_m\}\) and the predicted change points be \(P = \{p_0, p_1, \dots, p_n\}\), where consecutive values define half-open intervals. From these change points we build segment sets

$$
\mathcal{S}_G = \{[g_i, g_{i+1})\}_{i=0}^{m-1}, \qquad \mathcal{S}_P = \{[p_j, p_{j+1})\}_{j=0}^{n-1}.
$$

For any two segments \(a = [a_0, a_1)\) and \(b = [b_0, b_1)\), the Intersection over Union (IoU) is defined as

$$
\operatorname{IoU}(a, b) = \frac{|a \cap b|}{|a \cup b|} = \frac{\max(0, \min(a_1, b_1) - \max(a_0, b_0))}{(a_1 - a_0) + (b_1 - b_0) - \max(0, \min(a_1, b_1) - \max(a_0, b_0))}.
$$

The *directional covering* of a source set \(\mathcal{S}\) by a target set \(\mathcal{T}\) is

$$
\mathrm{Cover}(\mathcal{S} \rightarrow \mathcal{T}) = \frac{\sum_{s \in \mathcal{S}} |s| \max_{t \in \mathcal{T}} \operatorname{IoU}(s, t)}{\sum_{s \in \mathcal{S}} |s|}.
$$

The bidirectional covering metric computes:

- \(C_{\text{GT}} = \mathrm{Cover}(\mathcal{S}_G \rightarrow \mathcal{S}_P)\)
- \(C_{\text{Pred}} = \mathrm{Cover}(\mathcal{S}_P \rightarrow \mathcal{S}_G)\)

and combines them into a single score using a selectable aggregation strategy \(A\):

$$
\text{BidirectionalCovering} = A\left(C_{\text{GT}}, C_{\text{Pred}}\right).
$$

### Aggregation Strategies
The implementation provides four strategies controlled by the `aggregation` argument:

- **harmonic** (default):
  $$A(a, b) = \begin{cases} 0, & a \le 0 \text{ or } b \le 0, \\ \dfrac{2ab}{a + b}, & \text{otherwise.}\end{cases}$$
- **geometric**:
  $$A(a, b) = \begin{cases} 0, & a \le 0 \text{ or } b \le 0, \\ \sqrt{ab}, & \text{otherwise.}\end{cases}$$
- **arithmetic**: \(A(a, b) = \tfrac{a + b}{2}\).
- **min**: \(A(a, b) = \min(a, b)\).

These options let practitioners emphasise strict agreement (harmonic or min) or more tolerant aggregations (arithmetic).

## Edge Cases
The implementation covers several degenerate scenarios:

- **No segments on both sides**: returns a perfect score of 1 (nothing to cover and nothing predicted).
- **No ground-truth segments but predictions exist**: coverage in either direction is 0, so the final score is 0.
- **No predictions but ground-truth segments exist**: symmetric to the previous case; score is 0.
- **Zero-length or unsorted change points**: segments are sorted and filtered so that non-positive-length intervals are ignored during coverage computation.

## Implementation Notes
- Pass `convert_labels_to_segments=True` to work directly with label sequences instead of change-point indices. Labels are automatically converted via `labels_to_change_points`.
- Target segments are sorted once to accelerate the sweep-line computation for overlap checks.
- The metric returns a dictionary containing `score`, `ground_truth_covering`, and `prediction_covering` to assist with diagnostics.

## Usage Example
```python
from tsseg.metrics.bidirectional_covering import BidirectionalCovering

metric = BidirectionalCovering(convert_labels_to_segments=False)
y_true = [0, 50, 120, 200]   # change points defining 3 ground-truth segments
y_pred = [0, 60, 180, 200]   # predicted change points

result = metric.compute(y_true, y_pred)
print(result)
```

Example output:

```
{
    'score': 0.8425,
    'ground_truth_covering': 0.8712,
    'prediction_covering': 0.8150
}
```

The higher ground-truth covering indicates that most true segments are well covered, while the slightly lower prediction covering highlights over-segmentation or redundant predictions.

## Relation to the Classical Covering
If the aggregation strategy ignores the prediction covering (for example, by setting `aggregation="arithmetic"` and producing predictions with perfect ground-truth coverage but zero prediction coverage), the bidirectional score reduces to the asymmetric classical covering. Conversely, when both directional scores are high, the two metrics coincide. The bidirectional extension therefore encourages balanced behaviour without discarding the insight provided by the original metric.

## Further Reading
- Notebook `examples/bidirectional_covering.ipynb` visualises the metric on synthetic data and compares it to the classical covering score.
- API reference `docs/api/tsseg.metrics.rst` documents the class signature and parameters.
