tsseg.algorithms.amoc package
=============================

AMOC (At Most One Change) — single change point detection.

Description
-----------

AMOC searches for the single breakpoint that minimises the total sum of squared
errors (SSE) on either side of the split.  It is the foundational building block
used internally by multi-change detectors such as Binary Segmentation and PELT,
which repeatedly apply the single-change solver on sub-segments of the signal.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n\,d)` time, :math:`O(1)` extra memory

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``min_size``
     - int
     - ``5``
     - Minimum number of samples required on each side of the breakpoint.
   * - ``axis``
     - int
     - ``0``
     - Axis representing time in the input array.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import AmocDetector

   detector = AmocDetector(min_size=10)
   labels = detector.fit_predict(X)

**Implementation:** Clean-room reimplementation of the classical SSE-based single
change point criterion.  Inspired by the R ``changepoint`` package (GPL >= 2), no R
code reused.  *Origin: new code.*

**Reference:** Killick, Fearnhead & Eckley (2012), JASA.

Submodules
----------

tsseg.algorithms.amoc.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.amoc.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.amoc
   :show-inheritance:
