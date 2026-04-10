tsseg.algorithms.binseg package
===============================

Binary Segmentation (BinSeg) — fast sequential change point detection.

Description
-----------

Binary segmentation is a greedy, top-down procedure: it first detects a single
change point in the full signal, then splits the series at that point and
repeats the operation on each sub-signal until a stopping criterion is met.

The benefits include low complexity — :math:`O(C\,n\log n)` where *n* is the
number of samples and *C* the cost of evaluating the cost function on one
sub-signal — and the ability to work whether the number of regimes is known
beforehand or not.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(C\,n\log n)`

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
     - Ruptures cost model (``"l2"``, ``"l1"``, ``"rbf"``, ``"linear"``, ``"normal"``, ``"ar"``).
   * - ``min_size``
     - int
     - ``2``
     - Minimum segment length.
   * - ``jump``
     - int
     - ``5``
     - Sub-sampling factor for candidate breakpoints.
   * - ``penalty``
     - float / None
     - ``10``
     - Penalty value (BIC-style).  Mutually exclusive with ``n_cps`` / ``epsilon``.
   * - ``epsilon``
     - float / None
     - ``None``
     - Reconstruction-error tolerance.  Mutually exclusive with ``n_cps`` / ``penalty``.
   * - ``custom_cost``
     - BaseCost / None
     - ``None``
     - Pre-instantiated ruptures cost object.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Additional keyword arguments forwarded to the cost factory.
   * - ``axis``
     - int
     - ``0``
     - Axis representing time.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import BinSegDetector

   detector = BinSegDetector(model="l2", n_cps=3)
   labels = detector.fit_predict(X)

   # Unknown number of change points — use a penalty
   detector = BinSegDetector(model="l2", penalty=10)
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.

**Reference:** Bai (1997), Econometric Theory; Fryzlewicz (2014), The Annals of
Statistics.

Submodules
----------

tsseg.algorithms.binseg.detector module
---------------------------------------

.. automodule:: tsseg.algorithms.binseg.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.binseg
   :show-inheritance:
