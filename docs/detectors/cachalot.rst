Cachalot
========

Cachalot (CAtching CHAnge-points with LassO), also known as TV Lasso, detects
changes in the level of a piecewise-constant signal. It first selects
candidate change points along a Lasso path, then refines the segmentation
with dynamic programming restricted to those candidates. The number of
change points is chosen by the relative reduction in residual sum of squares.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate method; multivariate input is processed independently
  by channel
| **Complexity:** :math:`O(n K^2 + K^3)` for a candidate path of length *K*

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``max_candidates``
     - int
     - ``50``
     - Maximum number of Lasso-path candidate change points per channel,
       capped at ``n_timepoints - 1``.
   * - ``nu``
     - float
     - ``0.01``
     - Relative-improvement threshold. Select the first *k* such that
       ``J(k + 1) / J(k) >= 1 - nu``.
   * - ``axis``
     - int
     - ``0``
     - Axis of ``X`` that represents time.

The paper presents a univariate method. For multivariate input, each channel
gets its own Lasso path and reduced dynamic program; the detector returns the
union of their selected change points. This is a channel-wise extension, not a
shared-change-point model such as group LARS. On the nine four-channel MoCap
trials bundled with tsseg, the union raised macro F1 from 0.357 (mean of the
four single-channel scores) to 0.404 using a 1% matching margin; macro
precision and recall were 0.304 and 0.655. The mean per-trial best-channel
F1, an oracle reference, was 0.576. No predicted change point was exactly
shared by all four channels. This supports the union only as a high-recall
channel-wise extension, not as evidence of a shared multivariate model.

The paper's toy example reports candidate locations and an objective table,
but omits the sampled noise vector and random seed. Its Monte Carlo tables
also omit seeds, so those exact random realizations cannot be regenerated.
The paper reports that its toy objective table selects four change points at
``(30, 50, 70, 90)`` with ``nu=0.05``.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import CachalotDetector

   detector = CachalotDetector(max_candidates=50, nu=0.01)
   change_points = detector.fit_predict(X)

**References:** Harchaoui, Z. and Lévy-Leduc, C., *Catching Change-points
with Lasso*, Advances in Neural Information Processing Systems 20, 2007;
Harchaoui, Z. and Lévy-Leduc, C., *Multiple Change-Point Estimation With a
Total Variation Penalty*, Journal of the American Statistical Association
105(492):1480-1493, 2010.

API reference
-------------

.. automodule:: tsseg.algorithms.cachalot.detector
   :members:
   :show-inheritance:
   :undoc-members:
