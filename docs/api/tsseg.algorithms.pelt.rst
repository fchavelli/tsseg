tsseg.algorithms.pelt package
=============================

PELT — Pruned Exact Linear Time change point detection.

Description
-----------

PELT solves the penalised change point optimisation problem exactly by
pruning the search space with a dynamic-programming rule.  Under mild conditions
on the change point distribution, the average complexity is
:math:`O(C\,K\,n)` — linear in the number of samples — where *K* is the number
of change points and *C* the cost function complexity.  In practice, ``"l2"``
cost models are significantly faster than linear or autoregressive ones.

Key tuning levers:

- ``penalty`` — higher values produce fewer change points.
- ``min_size`` — minimum distance between change points.
- ``jump`` — grid step for admissible positions.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(C\,K\,n)` average

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
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
   * - ``penalty``
     - float
     - ``10.0``
     - Penalty threshold for the PELT stopping criterion.
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

   from tsseg.algorithms import PeltDetector

   detector = PeltDetector(model="l2", penalty=10)
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.

**Reference:** Killick, Fearnhead & Eckley (2012), *Optimal detection of
changepoints with a linear computational cost*, JASA.

Submodules
----------

tsseg.algorithms.pelt.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.pelt.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.pelt
   :show-inheritance:
