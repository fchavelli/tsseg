BottomUp
========

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
   * - ``pen_scale``
     - str / None
     - ``None``
     - ``"bic"``: ``penalty`` is a coefficient on :math:`\log(n)\,u`, *u* the cost
       of one sample of the whole signal: *d* for ``"l2"`` on unit-variance
       channels (the BIC penalty of one more segment), the mean absolute
       deviation for ``"l1"``, 1 for the kernel costs. One value then applies
       across lengths, dimensions and costs.
   * - ``epsilon``
     - float / None
     - ``None``
     - Reconstruction budget.  Mutually exclusive with ``n_cps`` / ``penalty``.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Extra parameters passed to the cost function.
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

   from tsseg.algorithms import BottomUpDetector

   detector = BottomUpDetector(model="l2", n_cps=3)
   change_points = detector.fit_predict(X)

   # Penalty-based stopping
   detector = BottomUpDetector(model="l2", penalty=10)
   change_points = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.  With the
``"l1"``, ``"l2"``, ``"rbf"`` or ``"cosine"`` cost and numba installed, the
segment costs are computed in numba: the kernel costs without the
:math:`n \times n` Gram matrix, ``"l1"`` and ``"l2"`` bit for bit as numpy
computes them. Merge gains closer than :math:`10^{-9}` of the segments' costs
are tied, and the leftmost merge is done first, on both backends.

**Reference:** Keogh, Chu, Hart & Pazzani (2001), ICDM; Fryzlewicz (2007), JASA.

API reference
-------------

.. automodule:: tsseg.algorithms.bottomup.detector
   :members:
   :show-inheritance:
   :undoc-members:
