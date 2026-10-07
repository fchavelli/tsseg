KCPD
====

KCPD — Kernel Change Point Detection.

.. warning::
   **Deprecated.** ``KCPDDetector`` is not a method of its own: it runs the solvers of
   :doc:`pelt` (with ``pen``) and :doc:`dynp` (with ``n_cps``) on a kernel
   cost, and will be removed in a future release. Use
   ``PeltDetector(model="rbf", penalty=pen, jump=1)`` or
   ``DynpDetector(model="rbf", n_cps=k, jump=1)`` (``model="l2"`` for the
   linear kernel). For ``decimation=k``, pass ``X[::k]`` and multiply the
   change points by ``k``; for ``pen_scale="bic"``, use a penalty of
   ``pen * log(n) * d``.

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
   change_points = detector.fit_predict(X)

   # Unknown K — use a penalty
   detector = KCPDDetector(kernel="rbf", pen=10)
   change_points = detector.fit_predict(X)

**Implementation:** vendored from ruptures v1.1.8, whose ``KernelCPD`` is a C
implementation of PELT and DynP for kernel costs. tsseg runs its vendored
Python ``Pelt`` / ``Dynp`` instead, or their numba backend (``backend``), which
for the ``rbf`` and ``cosine`` kernels transcribes that C code. BSD 2-Clause.

**Reference:** Celisse, Marot, Pierre-Jean & Rigaill (2018), Computational
Statistics and Data Analysis; Arlot, Celisse & Harchaoui (2019), JMLR.

API reference
-------------

.. automodule:: tsseg.algorithms.kcpd.detector
   :members:
   :show-inheritance:
   :undoc-members:
