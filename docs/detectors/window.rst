Window
======

Window — sliding-window discrepancy change point detection.

Description
-----------

Window-based detection slides two adjacent windows along the signal and computes
a *discrepancy* measure :math:`d(\cdot,\cdot)` derived from the cost function:

.. math::

   d(y_{u..v},\,y_{v..w}) = c(y_{u..w}) - c(y_{u..v}) - c(y_{v..w})

When both windows fall inside the same segment their statistical properties are
similar and the discrepancy is low.  When they straddle a change point the
discrepancy spikes.  A sequential peak search on the discrepancy curve produces
the final change points.

Complexity is :math:`O(n\,w)` where *w* is the window width — making Window one
of the fastest methods.  The ``jump`` parameter further speeds up prediction at
the expense of positional precision.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n\,w)`

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``width``
     - int
     - ``100``
     - Width of each sliding window (in samples).
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points.
   * - ``pen``
     - float / None
     - ``None``
     - Penalty threshold.
   * - ``pen_scale``
     - str / None
     - ``None``
     - ``"bic"``: ``pen`` is a coefficient on :math:`\log(n)\,u`, *u* the cost
       of one sample of the whole signal: *d* for ``"l2"`` on unit-variance
       channels (the BIC penalty of one more segment), the mean absolute
       deviation for ``"l1"``, 1 for the kernel costs. One value then applies
       across lengths, dimensions and costs.
   * - ``epsilon``
     - float / None
     - ``None``
     - Reconstruction-error tolerance.
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
     - Sub-sampling factor for candidates.
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

   from tsseg.algorithms import WindowDetector

   detector = WindowDetector(width=50, model="l2", n_cps=3)
   labels = detector.fit_predict(X)

   # Penalty-based stopping
   import numpy as np
   detector = WindowDetector(width=50, model="l2", pen=np.log(n) * d * sigma**2)
   labels = detector.fit_predict(X)

**Implementation:** Vendored from ruptures v1.1.8.  BSD 2-Clause.  With the
``"l1"``, ``"l2"``, ``"rbf"`` or ``"cosine"`` cost and numba installed, the
window scores and segment costs are computed in numba: the kernel costs without
the :math:`n \times n` Gram matrix, ``"l1"`` and ``"l2"`` bit for bit as numpy
computes them. Scores closer than :math:`10^{-9}` of the windows' costs are
tied: a peak must exceed its neighbours by more than that, and the latest of
tied peaks is taken first, on both backends. Peaks are searched as upstream
searches them (``scipy.signal.argrelmax`` with ``mode="wrap"``); until 0.2.0 the
neighbourhood of a score was cut at the ends of the signal instead, which could
add a change point next to the first or last window.

**Reference:** Basseville & Nikiforov (1993), *Detection of Abrupt Changes*,
Prentice Hall.

API reference
-------------

.. automodule:: tsseg.algorithms.window.detector
   :members:
   :show-inheritance:
   :undoc-members:
