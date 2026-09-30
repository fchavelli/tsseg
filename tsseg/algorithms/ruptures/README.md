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
- Change: every solver treats values closer than `utils.TIE_RTOL` (1e-9) of the
  costs involved as equal, and decides between them by position, as upstream
  decides between exactly equal values: first candidate (`Pelt`, `Dynp`), last
  change point of a segment and first segment (`Binseg`), leftmost merge
  (`BottomUp`), no peak on a plateau and latest tied peak first (`Window`); a
  value tied with `pen` or `epsilon` stops, and `Pelt` keeps a start tied with
  its pruning bound. The gap follows the scale of the signal
  (`utils.tie_unit`, the cost of one sample)
- Change: `Window` searches its peaks as upstream does, `argrelmax(mode="wrap")`
  (`utils.argrelmax_1d` used to cut the neighbourhood of a score at the ends,
  and could return a change point next to the first or last window)
- Change: the costs convert the signal to float64 in `fit` (upstream computes
  `l1` and `l2` in the input's dtype, and `CostCosine` failed on integers)
- Addition: `accel/`, a numba backend of the solvers for the `l1`, `l2`, `rbf`
  and `cosine` costs (`backend="auto"`, the default, uses it when numba is
  installed): the whole search of `Pelt` and `Dynp`, for the kernel costs a
  transcription of the C solvers of upstream `KernelCPD`
  (`ekcpd_pelt_computation.c`, `ekcpd_computation.c`) that keeps their `jump`,
  pruning and unclipped kernels; the splits of `Binseg`, by sweeps; the segment
  costs of `BottomUp` and `Window`. The kernel costs never build the n x n Gram
  matrix, and the `l1` and `l2` costs of one segment sum in numpy's order, bit
  for bit. The `l1` and `l2` sweeps run on the channels far from 0
  (|median| > 4 std) shifted by their median, so that a signal at 1e8 keeps its
  digits. Both backends return the same segmentation
- Source: https://github.com/deepcharles/ruptures
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)
- Licence file: `LICENSE` in this directory
