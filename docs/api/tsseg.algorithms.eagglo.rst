tsseg.algorithms.eagglo package
===============================

E-Agglo — energy-based agglomerative change point detection.

Description
-----------

E-Agglo is a non-parametric, hierarchical agglomerative algorithm for detecting
multiple change points in multivariate time series.  Neighbouring segments are
sequentially merged when the merge maximises a goodness-of-fit statistic based
on energy distances.  Unlike classical agglomerative clustering, this procedure
preserves the temporal ordering.

A divergence parameter :math:`\alpha\in(0,2]` controls the distance exponent.
An optional penalty function can regularise against over-segmentation.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n^{2})`
| **Requires:** numba

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``member``
     - array-like / None
     - ``None``
     - Initial cluster membership. ``None`` = one cluster per point.
   * - ``alpha``
     - float
     - ``1.0``
     - Divergence exponent in :math:`(0, 2]`.
   * - ``penalty``
     - str / callable / None
     - ``None``
     - Penalty function (``"len_penalty"``, ``"mean_diff_penalty"`` or callable).

Usage
-----

.. code-block:: python

   from tsseg.algorithms import EAggloDetector

   detector = EAggloDetector(alpha=1.0)
   labels = detector.fit_predict(X)

**Implementation:** Adapted from aeon.  BSD 3-Clause.

**Reference:** Matteson & James (2014), *A Nonparametric Approach for Multiple
Change Point Analysis of Multivariate Data*, JASA.

Submodules
----------

tsseg.algorithms.eagglo.detector module
---------------------------------------

.. automodule:: tsseg.algorithms.eagglo.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.eagglo
   :members:
   :show-inheritance:
   :undoc-members:
