tsseg.algorithms.dynp package
=============================

Dynamic Programming (DynP) — exact optimal segmentation.

Description
-----------

Dynamic programming finds the *exact* minimum of the sum of segment costs by
enumerating all admissible partitions.  The user **must** specify the number of
change points in advance (consider PELT or BinSeg when this is unknown).

Complexity is :math:`O(C\,K\,n^{2})` where *K* is the number of change points,
*n* the sample count and *C* the cost-function complexity.  To reduce runtime,
increase ``min_size`` and ``jump``:

- ``min_size`` controls the minimum distance between change points.
- ``jump`` controls the grid of admissible positions (only multiples of ``jump``).

| **Type:** change point detection
| **Supervision:** semi-supervised (``n_cps`` required)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(C\,K\,n^{2})`

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
     - int
     - ``1``
     - Number of change points to detect (>= 0).
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
     - Grid step for candidate positions.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Extra arguments for the cost function.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import DynpDetector

   detector = DynpDetector(n_cps=3, model="l2")
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.

**Reference:** Auger & Lawrence (1989), Bulletin of Mathematical Biology.

Submodules
----------

tsseg.algorithms.dynp.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.dynp.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.dynp
   :show-inheritance:
