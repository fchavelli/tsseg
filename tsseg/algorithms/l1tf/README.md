# l1 trend filtering

Change point detection by penalised piecewise-linear fitting: the estimated
trend minimises the objective of Kim, Koh, Boyd and Gorinevsky (2009, eq. 5),

    minimize (1/2)||y - x||^2 + lambda ||D x||_1,

with D the second-order difference matrix (first row `[1 -2 1 0 ... 0]`).
The trend is exactly linear between its kinks, and the kinks — the nonzero
second differences of the estimate — are the change points. Unlike H-P
filtering (the paper's section 2 comparison, reproduced on every series as
`lambda_max_`), the number of kinks is a *finite* consequence of `penalty`:
above `||(D D^T)^{-1} D y||_inf` (eq. 7, `lambda_max_`) the estimate is the
best affine fit.

## Key properties

- Type: change point detection
- Fully unsupervised (penalty-based)
- Univariate and multivariate (union of the per-channel kinks)
- Convex problem, unique minimiser; the number of change points follows from
  `penalty`, the kinks of a piecewise-linear trend are detected as changes
  of slope (no level-shift-only detection: a pure mean shift needs two kinks
  and a plateau between them)

## Implementation

- The problem is solved on its dual (paper section 5.2, eq. 13), the
  box-constrained QP `min (1/2) nu' H nu - nu' Dy s.t. |nu| <= lambda` with
  `H = D D'` (5-diagonal Toeplitz, bands 6, -4, 1), by
  [OSQP](https://osqp.org) (Apache-2.0), an optional dependency
  (`tsseg[l1tf]`), then by an exact active-set polish of our own: the free
  block of `(D D') nu = D y` is solved exactly once the saturated
  coordinates are fixed, and all KKT conditions of the box-QP are checked
  (feasibility, sign of the second differences along the bounds), which
  certifies the unique optimum. Above `lambda_max` (eq. 7) the dual is the
  banded unconstrained solve directly. The kink set is read off the dual
  (paper eq. 12: `y - x = D' nu`, `|nu_t| = lambda`) together with the
  second differences, which handles the degenerate coordinates that touch
  the box with an exactly zero second difference.
- Solver departure: the paper uses a specialized interior-point method
  (its section 6); the strictly convex problem has a unique minimiser, which
  OSQP returns up to its tolerances. Validated against CVXPY/CLARABEL on the
  paper's S&P 500 example and on the paper's properties (6) and (7)
  (`notes/l1tf/`).
- Multivariate: the "sum of l1 norms" branch of the paper's section 7.5 —
  the channels are filtered separately with the same `penalty`, the change
  points are the union of the channel kinks. The section's group penalty
  (common kinks, sum of the l2 norms of the second differences) needs a
  conic (second-order cone) solver; it is not implemented.
- Milder penalties detect more kinks: `knot_tol` (relative threshold on the
  second differences) separates the kinks from the solver noise; defaults
  calibrated against the OSQP/CLARABEL agreement in `notes/l1tf/`.

## Citation

```bibtex
@article{kim2009l1,
  title   = {{l1} Trend Filtering},
  author  = {Kim, Seung-Jean and Koh, Kwangmoo and Boyd, Stephen and
             Gorinevsky, Dimitry},
  journal = {SIAM Review},
  volume  = {51},
  number  = {2},
  pages   = {339--360},
  year    = {2009}
}
```
