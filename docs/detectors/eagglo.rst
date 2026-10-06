E-Agglo
=======

E-Agglo — energy-based agglomerative change point detection.

Description
-----------

E-Agglo is a non-parametric, hierarchical agglomerative algorithm for detecting
multiple change points in multivariate time series.  Adjacent segments are
merged greedily to maximise a goodness-of-fit statistic, the sum of the energy
divergences between adjacent segments, and the segmentation with the largest
statistic along the merging sequence is returned.  Unlike classical
agglomerative clustering, this procedure preserves the temporal ordering.

A divergence parameter :math:`\alpha\in(0,2]` controls the distance exponent
(:math:`\alpha = 2` only compares means).  From one cluster per point
(``member=None``, the default) the statistic is usually maximised by a very
fine segmentation: start from blocks of a few samples (``member``) and/or set a
penalty.  The method assumes independent observations and can over-segment
serially dependent series whatever the initial partition.

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
     - Initial cluster of each point (sorted labels, contiguous blocks). ``None`` = one cluster per point.
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

   import numpy as np
   from tsseg.algorithms import EAggloDetector

   member = np.arange(len(X)) // 10  # initial blocks of 10 samples
   detector = EAggloDetector(alpha=1.0, member=member)
   change_points = detector.fit_predict(X)

**Implementation:** Adapted from aeon.  BSD 3-Clause.

**Reference:** Matteson & James (2014), *A Nonparametric Approach for Multiple
Change Point Analysis of Multivariate Data*, JASA.

API reference
-------------

.. automodule:: tsseg.algorithms.eagglo.detector
   :members:
   :show-inheritance:
   :undoc-members:
