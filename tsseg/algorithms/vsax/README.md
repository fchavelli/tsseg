# Variable-length SAX Detector

This module implements a lightweight baseline for **state detection** based on a
variable-length Symbolic Aggregate approXimation (SAX) representation. It
segments the input signal into pieces that minimise a reconstruction error while
allowing the segment length to adapt within a user-defined range.

## Key ideas

- **Z-normalisation** of the incoming series for robustness across sensors.
- **Piecewise Aggregate Approximation (PAA)** to summarise segments.
- **Symbol assignment** using SAX breakpoints, shared across dimensions.
- **Dynamic programming** to pick a segmentation that balances fidelity and
  model complexity via a simple penalty term.

## Parameters

- `alphabet_size` – number of SAX symbols (3–10 supported).
- `paa_segments` – maximum number of PAA frames used inside each segment.
- `min_segment_length` / `max_segment_length` – bounds on admissible segment
  duration (in time steps).
- `num_lengths` – number of candidate segment lengths explored between the
  bounds.
- `penalty` – per-segment penalty controlling the amount of change-points.

## Outputs

The detector returns a dense array of **state labels** (integers) aligned with
input timestamps. Identical SAX words across the sequence map to the same state
identifier, which makes the method reproduce previously seen states when the
pattern reappears.
