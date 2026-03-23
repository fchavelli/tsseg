tsseg.algorithms.window package
===============================

Window — sliding-window discrepancy change point detection.

Description
-----------

Window-based detection slides two adjacent windows along the signal and computes
a *discrepancy* measure :math:`d(\cdot,\cdot)` derived from the cost function:

.. math::

   d(y_{u..v},\,y_{v..w}) = c(y_{u..w}) - c(y_{u..v}) - c(y_{v..w})

When both windows fall inside the same segment their statistical properties are
similar and the discrepancy is low.  When they straddle a change point the
discrepancy spikes.  A sequential peak search on the discrepancy curve produces
the final change points.

Complexity is :math:`O(n\,w)` where *w* is the window width — making Window one
of the fastest methods.  The ``jump`` parameter further speeds up prediction at
the expense of positional precision.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n\,w)`

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``width``
     - int
     - ``100``
     - Width of each sliding window (in samples).
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points.
   * - ``pen``
     - float / None
     - ``None``
     - Penalty threshold.
   * - ``epsilon``
     - float / None
     - ``None``
     - Reconstruction-error tolerance.
   * - ``model``
     - str
     - ``"l2"``
     - Ruptures cost model.
   * - ``min_size``
     - int
     - ``2``
     - Minimum segment length.
   * - ``jump``
     - int
     - ``5``
     - Sub-sampling factor for candidates.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Extra keyword arguments for the cost factory.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import WindowDetector

   detector = WindowDetector(width=50, model="l2", n_cps=3)
   labels = detector.fit_predict(X)

   # Penalty-based stopping
   import numpy as np
   detector = WindowDetector(width=50, model="l2", pen=np.log(n) * d * sigma**2)
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.

**Reference:** Basseville & Nikiforov (1993), *Detection of Abrupt Changes*,
Prentice Hall.

Submodules
----------

tsseg.algorithms.window.detector module
---------------------------------------

.. automodule:: tsseg.algorithms.window.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.window
   :members:
   :show-inheritance:
   :undoc-members:
