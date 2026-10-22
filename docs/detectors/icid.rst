iCID
====

iCID — Isolation Distributional Kernel Change Interval Detection.

Description
-----------

iCID detects *change intervals* by projecting the time series into a
high-dimensional distributional feature space (aNNEspace) and measuring
cosine-distance dissimilarity between consecutive sliding windows.  A
z-score-based threshold :math:`\alpha` determines whether a dissimilarity peak
constitutes a true change.

The method is non-parametric, does not assume Gaussianity, and automatically
selects the granularity through a list of ``psi`` (sub-sample) values.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n + n^{2})` for each value of ``psi_list``: the
  approximate entropy that selects ``psi`` compares all pairs of points of the
  score

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``window_size``
     - int
     - ``50``
     - Sliding window size for computing dissimilarity.
   * - ``alpha``
     - float
     - ``0.5``
     - Sensitivity factor (higher = less sensitive).
   * - ``t``
     - int
     - ``200``
     - Iterations for aNNEspace transformation.
   * - ``psi_list``
     - list[int] / None
     - ``[2,4,8,16,32,64]``
     - Sub-sample sizes controlling granularity.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ICIDDetector

   detector = ICIDDetector(window_size=100, alpha=0.3)
   change_points = detector.fit_predict(X)

**Implementation:** Python translation of the authors' MATLAB code
(`IsolationKernel/iCID <https://github.com/IsolationKernel/iCID>`_:
``aNNEspace``, ``point_score``, ``best_psi``, ``best_threshold``), which is not
shipped with tsseg.  GPLv3; a copy of the licence is in
``tsseg/algorithms/icid/LICENSE``.  The MATLAB code labels each point 0 or 1
against the threshold; returning the first point of each run of 1 as a change
point is specific to tsseg.  *Origin: adapted from the original MATLAB code.*

**Reference:** Cao, Zhu, Ting, Salim, Li, Yang & Li (2024), *Detecting Change
Intervals with Isolation Distributional Kernel*, JAIR 79, 273–306.

API reference
-------------

.. automodule:: tsseg.algorithms.icid.detector
   :members:
   :show-inheritance:
   :undoc-members:
