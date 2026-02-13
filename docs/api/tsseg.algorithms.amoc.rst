tsseg.algorithms.amoc package
=============================

Single change point detector that searches for the breakpoint minimising the total
sum of squared errors (SSE) on either side of the split. Foundational building block
for multi-change detectors such as Binary Segmentation and PELT.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** O(n d) time, O(1) extra memory

**Implementation:** Clean-room reimplementation of the classical SSE-based single
change point criterion. Inspired by the R ``changepoint`` package (GPL ≥ 2), no R
code reused. *Origin: new code.*

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
   :members:
   :show-inheritance:
   :undoc-members:
