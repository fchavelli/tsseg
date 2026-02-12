# VSAX (Variable-length SAX Detector)

Lightweight state detection baseline using variable-length Symbolic Aggregate
Approximation. Z-normalises the input, computes Piecewise Aggregate
Approximation (PAA) summaries, assigns SAX symbols via breakpoints, then uses
dynamic programming to find the segmentation that balances reconstruction
fidelity and model complexity.

## Key properties

- Type: state detection
- Semi-supervised (penalty controls number of segments)
- Univariate and multivariate
- Identical SAX words map to the same state label across the sequence

## Implementation

- Origin: new code
