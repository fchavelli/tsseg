# VSAX (Variable-length SAX Detector)

State detection baseline using variable-length Symbolic Aggregate
Approximation.  Z-normalises the input, computes Piecewise Aggregate
Approximation (PAA) summaries on each channel independently, assigns
**per-channel** SAX symbols via breakpoints, then uses dynamic programming
to find the segmentation that minimises reconstruction error plus an
additive penalty controlling model complexity.

Similar SAX symbols are merged into the same state via agglomerative
clustering on Hamming distance, avoiding the brittleness of exact matching.

Reconstruction cost is computed in O(1) per candidate segment using prefix
sums (cumulative sums), making the DP sweep efficient even for long series.

## Key properties

- Type: state detection
- Unsupervised / semi-supervised (adaptive breakpoints from fit data)
- Univariate and multivariate (per-channel SAX symbols)
- Similar SAX words are clustered into the same state label
- O(n × L) DP with O(1) per-segment cost via prefix sums

## Parameters of interest

| Parameter | Effect |
|-----------|--------|
| `penalty` | Controls number of segments (higher → fewer / longer) |
| `symbol_merge_threshold` | Hamming fraction for merging symbols (0 = exact, 0.2 = default) |
| `adaptive_breakpoints` | Learn breakpoints from data quantiles instead of Gaussian |
| `alphabet_size` | SAX resolution per channel |
| `paa_segments` | Temporal resolution of the SAX word |

## Implementation

- Origin: new code
