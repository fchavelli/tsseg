tsseg.algorithms.bottomup package
=================================

Bottom-Up segmentation — agglomerative merging of segments.

Description
-----------

Contrary to binary segmentation (top-down), bottom-up segmentation starts with a
fine initial partition along a regular grid and successively merges the two most
similar contiguous segments until a stopping criterion is met.  The procedure is
*generous*: it begins with many change points and removes the least significant
ones.

Complexity is :math:`O(n\log n)` where *n* is the number of samples, and the
algorithm supports any cost function available in the vendored ruptures library.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n\log n)`

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points.  When ``None``, uses ``penalty`` or ``epsilon``.
   * - ``model``
     - str
     - ``"l2"``
     - Ruptures cost model (``"l2"``, ``"l1"``, ``"rbf"``, etc.).
   * - ``min_size``
     - int
     - ``2``
     - Minimum segment length.
   * - ``jump``
     - int
     - ``5``
     - Grid step for candidate change points.
   * - ``penalty``
     - float / None
     - ``10``
     - Penalty value.  Mutually exclusive with ``n_cps`` / ``epsilon``.
   * - ``epsilon``
     - float / None
     - ``None``
     - Reconstruction budget.  Mutually exclusive with ``n_cps`` / ``penalty``.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Extra parameters passed to the cost function.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import BottomUpDetector

   detector = BottomUpDetector(model="l2", n_cps=3)
   labels = detector.fit_predict(X)

   # Penalty-based stopping
   detector = BottomUpDetector(model="l2", penalty=10)
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.

**Reference:** Keogh, Chu, Hart & Pazzani (2001), ICDM; Fryzlewicz (2007), JASA.

Submodules
----------

tsseg.algorithms.bottomup.detector module
-----------------------------------------

.. automodule:: tsseg.algorithms.bottomup.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.bottomup
   :members:
   :show-inheritance:
   :undoc-members:
