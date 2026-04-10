tsseg.algorithms.autoplait package
==================================

AutoPlait — parameter-free mining of co-evolving time sequences.

Description
-----------

AutoPlait is a fully automatic algorithm for segmenting and summarising large
collections of co-evolving time series.  It discovers an unknown number of
*regimes* (recurring patterns of different durations) by compressing the data
into a hierarchy of Hidden Markov Models.  Each regime is modelled as a separate
AR-HMM, and the algorithm uses a Minimum Description Length (MDL) criterion to
jointly determine the optimal number of regimes, their boundaries and their
internal structure — all without user-specified parameters.

The method operates in four steps:

1. **CutPointSearch** — locate candidate segment boundaries by fitting HMMs.
2. **CreateRegime** — build an AR-HMM for each candidate segment.
3. **MergeRegime** — greedily merge similar regimes to reduce the description
   length.
4. **Iterate** — repeat until convergence.

AutoPlait is linear in the input size and achieves near-perfect precision and
recall on the benchmarks reported in the original paper.

| **Type:** state detection
| **Supervision:** semi-supervised (requires ``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n)`

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
     - Number of change points.  Required if ground-truth ``y`` is not provided.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import AutoPlaitDetector

   detector = AutoPlaitDetector(n_cps=4)
   labels = detector.fit_predict(X)

.. note::
   AutoPlait relies on a compiled C binary (``autoplait_c``).  The binary is
   bundled in ``tsseg/c/autoplait/`` and must be built before first use.

**Implementation:** Adapted from the original C implementation by Matsubara et al.
BSD 3-Clause.

**Reference:** Matsubara, Sakurai & Faloutsos (2014), *AutoPlait: Automatic Mining
of Co-evolving Time Sequences*, SIGMOD.

Submodules
----------

tsseg.algorithms.autoplait.detector module
------------------------------------------

.. automodule:: tsseg.algorithms.autoplait.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.autoplait
   :show-inheritance:
