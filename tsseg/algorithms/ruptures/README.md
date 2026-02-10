# Vendored Ruptures Components

This directory contains a lightweight, pure-Python subset of the
[`ruptures`](https://github.com/deepcharles/ruptures) library that tsseg
uses internally. The goal is to remove the external dependency while
preserving core change point detection functionality. The code is
adapted from ruptures v1.1.8 and trimmed to the pieces required by the
existing detectors:

- shared estimator and cost base classes
- cost functions (`l1`, `l2`, `linear`, `rbf`, `cosine`, `normal`)
- utility helpers (peak detection, path reconstruction, sanity checks)
- detection algorithms (`Binseg`, `BottomUp`, `Dynp`, `Pelt`, `Window`)

Only limited refactoring was applied to align with the tsseg code style
and avoid optional SciPy dependencies. When updating this directory,
please keep a note of the upstream commit and review the LICENSE file in
ruptures to ensure continued compliance.
