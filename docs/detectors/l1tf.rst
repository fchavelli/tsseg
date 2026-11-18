l1 trend filtering
==================

l1 trend filtering (Kim, Koh, Boyd & Gorinevsky, 2009).

Description
-----------

On every channel, the detector estimates the piecewise-linear trend that
minimises the penalised objective of Kim, Koh, Boyd and Gorinevsky (2009,
eq. 5),

.. math:: \min_x \; \frac12\|y-x\|_2^2 + \lambda\|D x\|_1,

where :math:`D` is the second-order difference matrix. The kinks of the
estimate (its nonzero second differences) are reported as change points: a
piecewise-linear trend has an exact zero second difference between two kinks,
so the detector is naturally aimed at changes of slope (trend breaks).

The problem is convex but not smooth; it is solved through its dual, the
box-constrained QP of the paper (section 5.2, eq. 13),

.. math:: \min_\nu \; \tfrac12\nu^\top D D^\top \nu - \nu^\top D y
          \quad \text{s.t. } |\nu| \le \lambda,

by OSQP followed by an exact active-set polish: the free block of the
dual system is solved in closed form once the saturated coordinates are
identified, which certifies the KKT conditions of the unique optimum. The
values of interest of ``penalty`` depend on the scale of the series;
above :math:`\lambda_{max} = \|(D D^\top)^{-1} D y\|_\infty` (paper, eq. 7,
exposed as ``lambda_max_``) the estimate is the best affine fit and no
change point is reported.

Multivariate series use the "sum of l1 norms" extension of the paper
(section 7.5): each channel is filtered separately with the same penalty and
the change points are the union of the channel kinks. The group-penalty
variant of the same section (common kinks across channels, sum of l2 norms)
requires a conic solver and is not implemented.

| **Type:** change point detection
| **Supervision:** unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n)` memory per solve; the OSQP iterations are
  :math:`O(n)` each, and the polish is a sparse solve of the reduced system
| **Requires:** OSQP (:code:`pip install tsseg[l1tf]`)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 55

   * - Name
     - Type
     - Default
     - Description
   * - ``penalty``
     - float
     - ``1.0``
     - l1 penalty :math:`\lambda > 0` on the second differences of the
       trend; compare with ``lambda_max_``.
   * - ``knot_tol``
     - float
     - ``1e-3``
     - Tolerance of the kink tests (dual saturation
       ``|dual| >= (1 - knot_tol) * penalty`` and relative second
       difference).
   * - ``eps_abs``
     - float
     - ``1e-6``
     - Absolute tolerance of OSQP (steers the active-set search only; the
       polish certifies the solution).
   * - ``eps_rel``
     - float
     - ``1e-6``
     - Relative tolerance of OSQP.
   * - ``max_iter``
     - int
     - ``200000``
     - Maximum number of OSQP iterations.
   * - ``polish``
     - bool
     - ``True``
     - Whether OSQP polishes its solution.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Attributes
----------

``change_points_``
    Sorted indices of the kinks of the estimated trends.
``trend_``
    The estimated piecewise-linear trends, same shape as the input.
``second_diff_``
    The second differences of the estimated trends (exact zeros between
    kinks).
``lambda_max_``
    :math:`\|(D D^\top)^{-1} D y\|_\infty` over channels: above this penalty
    the estimate is the best affine fit (paper, eq. 7).

Usage
-----

.. code-block:: python

   import numpy as np
   from tsseg.algorithms import TrendFilteringDetector

   t = np.arange(500)
   x = 2.0 + 0.01 * t + 0.05 * np.maximum(t - 250, 0) + 0.1 * np.sin(t)
   detector = TrendFilteringDetector(penalty=5.0)
   change_points = detector.fit_predict(x)

**Implementation:** original reimplementation. The convex problem is solved
by `OSQP <https://osqp.org>`_ (Apache-2.0); its model setup follows the dual
formulation of the paper's section 5.2 without reusing any code of the
authors' MATLAB implementation (GPL-2, interfaces CVX). The solver differs
from the paper's specialized interior-point method (its section 6), but the
problem is strictly convex, so the minimiser is the same, and the KKT
solution is certified by the active-set polish.

**Reference:** Kim, S.-J., Koh, K., Boyd, S. and Gorinevsky, D. (2009),
*l1 Trend Filtering*, SIAM Review 51(2):339-360.

Validation
----------

The implementation reproduces the properties of the paper on its own grounds:
the objective values and trends match an independent CVXPY/Clarabel solve of
eq. (5) to 1e-8 (relative) on synthetic and real series, the kinks carry the
same second differences as the reference solve, :math:`\|y - x\|_\infty \le
4\lambda` holds (property 6), the estimate is the best affine fit above
:math:`\lambda_{max}` (eq. 7), and the estimate is equivariant when the data
and the penalty are scaled together. On the S&P 500 log-price series of the
paper's figure 2 (n=2000) the detected kink positions agree with an
independent solve of the same problem; the paper reports 8 change points at
:math:`\lambda = 100` where the exact KKT support has 12 (including kinks
with a second difference a hundred times smaller than the largest one, which
the paper's figure does not separate).

API reference
-------------

.. automodule:: tsseg.algorithms.l1tf.detector
   :members:
   :show-inheritance:
   :undoc-members:
