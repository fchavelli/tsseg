Ruptures (vendored)
===================

Lightweight subset of ruptures v1.1.8 used by BinSeg, BottomUp, DynP, KCPD, PELT,
Window. Contains base classes, cost functions, utilities and detection algorithms.
Not a detector itself.

The solvers have a numba backend (``ruptures.accel``, ``backend="auto"`` when
numba is installed) for the ``l1``, ``l2``, ``rbf`` and ``cosine`` costs: the
kernel costs never build the :math:`n \times n` Gram matrix, and both backends
return the same segmentation.

**Implementation:** Vendored from `ruptures <https://github.com/deepcharles/ruptures>`_.
BSD 2-Clause.
