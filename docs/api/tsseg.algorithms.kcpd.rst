tsseg.algorithms.kcpd package
=============================

KCPD — Kernel Change Point Detection.

Description
-----------

KCPD operates in a Reproducing Kernel Hilbert Space (RKHS) and detects
mean-shifts in the mapped signal.  The time series is implicitly mapped by a
kernel function :math:`k(\cdot,\cdot)` and the algorithm minimises:

.. math::

   V(t_1,\dots,t_K) = \sum_{k=0}^{K}\sum_{t=t_k}^{t_{k+1}-1}
     \|\phi(y_t) - \bar\mu_{t_k..t_{k+1}}\|_{\mathcal{H}}^2

When the number of changes *K* is known, the exact minimum is found by dynamic
programming; otherwise a penalised formulation (PELT) is used.

Available kernels: ``"linear"`` (L2 cost), ``"rbf"`` (Gaussian), ``"cosine"``.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
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
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points.  ``None`` = use ``pen``.
   * - ``pen``
     - float / None
     - ``10``
     - Penalty for the penalised optimisation.
   * - ``kernel``
     - str
     - ``"rbf"``
     - Kernel to use (``"linear"``, ``"rbf"``, ``"cosine"``).
   * - ``min_size``
     - int
     - ``2``
     - Minimum segment length.
   * - ``jump``
     - int
     - ``1``
     - Sub-sampling factor for candidates.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Extra arguments for the kernel cost.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import KCPDDetector

   detector = KCPDDetector(kernel="rbf", n_cps=3)
   labels = detector.fit_predict(X)

   # Unknown K — use a penalty
   detector = KCPDDetector(kernel="rbf", pen=10)
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8 (C implementation).  BSD
2-Clause.

**Reference:** Celisse, Marot, Pierre-Jean & Rigaill (2018), Computational
Statistics and Data Analysis; Arlot, Celisse & Harchaoui (2019), JMLR.

Submodules
----------

tsseg.algorithms.kcpd.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.kcpd.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.kcpd
   :members:
   :show-inheritance:
   :undoc-members:
