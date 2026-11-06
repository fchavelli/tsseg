CLaP
====

CLaP (Classification Label Profile) – classification-based state detection.

Description
-----------

CLaP extends :doc:`ClaSP <clasp>` from change point detection to *state
detection*. It first segments the series with ClaSP, then cross-validates a
time series classifier (e.g. ROCKET) on segment-labelled subsequences.
Segments with high confusion are iteratively merged, producing a set of state
labels that can be reused across non-contiguous time regions.

CLaP runs in three modes:

- **Unsupervised**: the number of segments is learned by ClaSP.
- **Guided**: provide ``n_segments`` or ``n_change_points``.
- **Exact**: provide pre-computed ``change_points``; ClaSP is not run.

| **Type:** state detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(K\,n\,(d\,n + K\log K))`, where *K* is the number of
  change points

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Name
     - Type
     - Default
     - Description
   * - ``window_size``
     - str / int
     - ``"suss"``
     - Window size or auto-detection method.
   * - ``classifier``
     - str
     - ``"rocket"``
     - aeon time series classifier: ``"rocket"``, ``"mrhydra"``
       (MultiRocket-Hydra), ``"weasel"`` (WEASEL 2.0), ``"quant"``, ``"rdst"``,
       ``"proximityforest"``, ``"freshprince"`` or ``"inception"`` (needs
       ``aeon[dl]``).
   * - ``merge_score``
     - str
     - ``"cgain"``
     - Score that decides whether two confused labels are merged:
       ``"cgain"`` (classification gain), ``"f1_score"``, ``"log_loss"``,
       ``"ami"``, ``"hamming_loss"`` or ``"roc_auc"``.
   * - ``n_splits``
     - int
     - ``5``
     - Cross-validation folds.
   * - ``n_jobs``
     - int
     - ``1``
     - Parallel jobs (``-1`` = all CPUs).
   * - ``sample_size``
     - int
     - ``1000``
     - Samples for classifier training.
   * - ``n_change_points``
     - int / None
     - ``None``
     - Number of change points (overrides ``n_segments``; ``0`` = one segment).
   * - ``change_points``
     - array-like / None
     - ``None``
     - Known change points (exact mode; ClaSP is not run).
   * - ``n_segments``
     - int / None
     - ``None``
     - Number of segments (guided mode).
   * - ``distance``
     - str
     - ``"znormed_euclidean_distance"``
     - Distance of ClaSP's k-NN: ``"znormed_euclidean_distance"``,
       ``"euclidean_distance"`` or ``"cinvariant_euclidean_distance"``. The
       z-normalised default misses a change of level alone (a mean shift in
       noise).
   * - ``random_state``
     - int
     - ``42``
     - Seed of CLaP's classifiers; ClaSP keeps the fixed seed of claspy
       (2357).

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ClapDetector

   # Guided: four segments, one state label per time point
   clap = ClapDetector(n_segments=4, classifier="rocket")
   states = clap.fit_predict(X)

**Implementation:** Adapted from original ClaSP code by Arik Ermshaus.  BSD
3-Clause.

**Reference:** Ermshaus, Schäfer & Leser (2024), *CLaP – State Detection from
Time Series*.

API reference
-------------

.. automodule:: tsseg.algorithms.clap.clap
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.clap.clap_detector
   :members:
   :show-inheritance:
   :undoc-members:
