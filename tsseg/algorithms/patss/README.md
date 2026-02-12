# PaTSS (Pattern-based Time Series Segmentation)

State detection algorithm designed to handle gradual state transitions.
Learns recurring patterns and assigns state labels, accommodating the fact that
real-world regime changes are not always instantaneous.

## Key properties

- Type: state detection
- Fully unsupervised
- Univariate and multivariate
- Handles gradual transitions between states

## Implementation

Adapted from the original PaTSS repository by the DTAI research group at
KU Leuven. The core logic lives in the `algorithms/`, `embedding/` and
`segmentation/` subdirectories.

- Origin: adapted from https://gitlab.kuleuven.be/u0143709/patss
- Licence: MIT (Copyright (c) 2023, KU Leuven, DTAI Research Group)
- Licence file: `LICENCE` in this directory

## Citation

```bibtex
@article{carpentier2024patss,
  title   = {{PaTSS}: Pattern-based Time Series Segmentation},
  author  = {Carpentier, Louis and others},
  year    = {2024}
}
```
