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
- Change: `Pelt` delays pruning by `min_size`, as in upstream
  [PR #383](https://github.com/deepcharles/ruptures/pull/383) (not merged yet)
- Change: `CostRbf` sets its median heuristic in `fit` without the Gram matrix,
  over the distances between distinct samples (rounding residues of the Gram's
  diagonal used to enter it), and builds the Gram from `pdist`, as upstream
- Addition: `ekcpd.py`, a numba transcription of the C solvers of upstream
  `KernelCPD` (`ekcpd_pelt_computation.c`, `ekcpd_computation.c`), run by `Pelt`
  and `Dynp` for the `rbf` and `cosine` costs; it keeps their `jump`, pruning and
  unclipped kernels, so both paths return the same segmentation
- Source: https://github.com/deepcharles/ruptures
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)
- Licence file: `LICENSE` in this directory
