tsseg.algorithms.icid package
=============================

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
   labels = detector.fit_predict(X)

**Implementation:** *Origin: new code.*

**Reference:** Cao, Ting, Liu, Cek & Angelova (2024), *A new framework for
change interval detection*, JAIR.

Submodules
----------

tsseg.algorithms.icid.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.icid.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.icid
   :members:
   :show-inheritance:
   :undoc-members:
