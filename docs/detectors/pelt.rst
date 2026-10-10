PELT
====

PELT — Pruned Exact Linear Time change point detection.

Description
-----------

PELT solves the penalised change point optimisation problem exactly by
pruning the search space with a dynamic-programming rule.  When the number of
change points grows linearly with the number of samples *n* (Killick et al.,
2012), the expected complexity is :math:`O(C\,n)` — linear in *n* and
independent of the number of change points — where *C* is the cost function
complexity; with few change points, little is pruned and the worst case is
:math:`O(C\,n^{2})`.  In practice, ``"l2"``
cost models are significantly faster than linear or autoregressive ones.

Key tuning levers:

- ``penalty`` — higher values produce fewer change points.
- ``min_size`` — minimum distance between change points.
- ``jump`` — grid step for admissible positions.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(C\,n)` expected, :math:`O(C\,n^{2})` worst case

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
   * - ``pen_scale``
     - str / None
     - ``None``
     - ``"bic"``: ``penalty`` is a coefficient on :math:`\log(n)\,u`, *u* the cost
       of one sample of the whole signal: *d* for ``"l2"`` on unit-variance
       channels (the BIC penalty of one more segment), the mean absolute
       deviation for ``"l1"``, 1 for the kernel costs. One value then applies
       across lengths, dimensions and costs.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Extra keyword arguments for the cost factory.
   * - ``backend``
     - str
     - ``"auto"``
     - Backend of the ``"l1"``, ``"l2"``, ``"rbf"`` and ``"cosine"`` costs:
       ``"auto"`` uses numba when it is installed (the kernel costs then in
       O(n) memory, without the :math:`n \times n` Gram matrix); ``"numba"``
       and ``"python"`` force either. Same segmentation on both paths.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import PeltDetector

   detector = PeltDetector(model="l2", penalty=10)
   change_points = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.  With the
``"l1"``, ``"l2"``, ``"rbf"`` or ``"cosine"`` cost and numba installed, the
search runs in numba. For the kernel costs (kernel change point detection,
ruptures' ``KernelCPD``) it transcribes ruptures' C code, which updates the
kernel sums sample by sample instead of storing the :math:`n \times n` Gram
matrix. Candidates closer than :math:`10^{-9}` of the costs involved are tied,
and the earliest change point wins, on both backends.

**Reference:** Killick, Fearnhead & Eckley (2012), *Optimal detection of
changepoints with a linear computational cost*, JASA.

API reference
-------------

.. automodule:: tsseg.algorithms.pelt.detector
   :members:
   :show-inheritance:
   :undoc-members:
