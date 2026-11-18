Cachalot
========

Cachalot (CAtching CHAnge-points with LassO) detects
changes in the level of a piecewise-constant signal. It first selects
candidate change points along a Lasso path, then refines the segmentation
with dynamic programming restricted to those candidates. The number of
change points is chosen by the relative reduction in residual sum of squares.
The total-variation Lasso supplies the candidates; Cachalot names the complete
three-stage procedure in the 2007 paper.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate method; multivariate input is processed independently
  by channel
| **Complexity:** :math:`O(n K + K^3)` time and :math:`O(n + K^2)` auxiliary memory per channel

Here *n* is the series length and ``K = min(max_candidates, n - 1)`` for
``n >= 2``. Channels are processed sequentially; per-channel processing
costs :math:`O(d (n K + K^3))` for *d* channels. Sorting the final union adds
an :math:`O(d K \log(d K))` bound. Total auxiliary memory, including stored
per-channel results and excluding the input array, is
:math:`O(n + d K^2)` (objective arrays retain their dynamic-programming
tables).

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

Usage
-----

.. code-block:: python

   from tsseg.algorithms import CachalotDetector

   detector = CachalotDetector(max_candidates=50, nu=0.01)
   change_points = detector.fit_predict(X)

**Implementation:** independent reimplementation of Section 3 of the 2007
paper, under AGPL-3.0-only. Centered, unstandardized step columns leave the
signal level unpenalized, following Eq. (4). The structured Lasso path uses
:math:`O(n K)` time per channel; prefix sums and reduced dynamic programming
use :math:`O(n + K^3)` time. Defaults follow Section 5.1.

Selection includes zero changes. A zero residual stops selection; if no
successive residual ratio reaches ``1 - nu``, all candidates are retained.
For multivariate input, each channel has its own path and model selection;
the result is the sorted union of the selected change points. Nearby
detections from different channels are kept separately, so the union has up
to ``min(n - 1, d * K)`` points.

**References:** Harchaoui, Z. and Lévy-Leduc, C.,
`Catching Change-points with Lasso <https://papers.neurips.cc/paper_files/paper/2007/file/e5841df2166dd424a57127423d276bbe-Paper.pdf>`_,
Advances in Neural Information Processing Systems 20, 2007 (algorithm source);
`Multiple Change-Point Estimation With a Total Variation Penalty <https://doi.org/10.1198/jasa.2010.tm09181>`_,
Journal of the American Statistical Association 105(492):1480-1493, 2010
(related journal article).

API reference
-------------

.. automodule:: tsseg.algorithms.cachalot.detector
   :members:
   :show-inheritance:
   :undoc-members:
