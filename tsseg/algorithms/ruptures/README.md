# Vendored Ruptures Components

Lightweight subset of ruptures v1.1.8 used internally by several
detectors (BinSeg, BottomUp, DynP, KCPD, PELT, Window). Contains base classes,
cost functions, utilities and detection algorithms.

## Contents

- `base/` -- base classes for cost functions and search methods
- `costs/` -- cost models: L1, L2, linear, RBF, cosine, normal
- `detection/` -- search algorithms: Binseg, BottomUp, Dynp, Pelt, Window
- `utils/` -- utility functions (e.g. pairwise distances)
- `exceptions.py` -- custom exception classes

## Key properties

- Not a detector itself; provides building blocks for other detectors
- API-compatible with the upstream ruptures package

## Implementation

Vendored from the ruptures library to avoid an external dependency and to allow
minor modifications.

- Origin: vendored from ruptures v1.1.8
- Source: https://github.com/deepcharles/ruptures
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)
- Licence file: `LICENSE` in this directory
