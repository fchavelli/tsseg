# Window (Window-Based Segmentation)

Sliding-window change point detection. Computes a discrepancy (gain) score by
comparing the distributions inside two adjacent windows as they slide along the
signal. Peaks in the score profile indicate change points.

## Key properties

- Type: change point detection
- Semi-supervised (`n_cps`) or unsupervised (`penalty` or `epsilon`)
- Supports multiple cost models: L2, L1, RBF, linear, normal, cosine
- Univariate and multivariate

## Implementation

Wraps the vendored ruptures `Window` solver.

- Origin: vendored from ruptures v1.1.8
- Costs `l1`, `l2`, `rbf`, `cosine`: with numba installed (`backend="auto"`),
  the window scores and segment costs are computed in numba
  (`ruptures/accel/`): the kernel costs without the n x n Gram matrix, `l1` and
  `l2` bit for bit as numpy computes them. It returns the same change points as
  the Python path (`backend="python"`).
- Ties: scores closer than 1e-9 of the windows' costs are tied: a peak must
  exceed its neighbours by more than that, and the latest of tied peaks is
  taken first, as upstream among exact ties.
- Peaks: a score is a peak if it exceeds its neighbours on each side, the score
  array wrapping around at its ends, as upstream (`argrelmax(mode="wrap")`);
  until 0.2.0 the neighbourhood was cut at the ends, which let a score falling
  away from the first or last window be a peak, hence a change point there.
- Licence: BSD 2-Clause (Copyright (c) 2017-2023, Charles Truong, Laurent Oudre, Nicolas Vayatis)

## Citation

```bibtex
@book{basseville1993detection,
  title     = {Detection of Abrupt Changes: Theory and Application},
  author    = {Basseville, Mich{\`e}le and Nikiforov, Igor V.},
  year      = {1993},
  publisher = {Prentice Hall}
}
```
