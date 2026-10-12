AutoPlait
=========

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
| **Supervision:** unsupervised
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
     - Deprecated and ignored.  AutoPlait chooses the number of segments
       itself and the bundled binary has no guided mode, so this value has
       never changed the output.  Passing it emits a ``FutureWarning`` at
       ``fit``; it will be removed in a future release.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import AutoPlaitDetector

   detector = AutoPlaitDetector()
   labels = detector.fit_predict(X)

``labels`` holds, for each time point, the number of the AutoPlait regime its
segment belongs to: segments of the same regime share a label.

.. note::
   AutoPlait relies on a compiled C binary (``autoplait``).  Its sources are in
   ``c/autoplait/`` of the source repository (not in the wheel); build them with
   ``make autoplait`` there, or point ``AUTOPLAIT_DIR`` to the directory that
   holds the binary.

   The binary counts the time points of its input with a line buffer of 254
   bytes: a line of more than 252 characters counts as several time points.
   The wrapper writes each value as a float32 with 9 significant digits (the
   precision at which the binary reads it), so that series of up to 15 channels
   always fit; a series whose rows do not fit (the 76 channels of JIGSAWS, for
   instance) raises ``ValueError`` before the binary runs.

**Implementation:** Adapted from the original C implementation by Matsubara et al.
BSD 3-Clause.

**Reference:** Matsubara, Sakurai & Faloutsos (2014), *AutoPlait: Automatic Mining
of Co-evolving Time Sequences*, SIGMOD.

API reference
-------------

.. automodule:: tsseg.algorithms.autoplait.detector
   :members:
   :show-inheritance:
   :undoc-members:
