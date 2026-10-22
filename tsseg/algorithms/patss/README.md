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
KU Leuven. The core logic lives in `algorithms/PaTSS_perso.py` and in the
`embedding/` subdirectory.

- Origin: adapted from https://gitlab.kuleuven.be/u0143709/patss
- Licence: MIT (Copyright (c) 2023, KU Leuven, DTAI Research Group)
- Licence file: `LICENCE` in this directory

## Citation

```bibtex
@inproceedings{carpentier2024patss,
  title     = {Pattern-based Time Series Semantic Segmentation with Gradual
               State Transitions},
  author    = {Carpentier, Louis and Feremans, Len and Meert, Wannes and
               Verbeke, Mathias},
  booktitle = {Proceedings of the 2024 SIAM International Conference on Data
               Mining (SDM)},
  pages     = {316--324},
  year      = {2024},
  doi       = {10.1137/1.9781611978032.36}
}
```
