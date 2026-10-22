CLaP / ClaSP
============

ClaSP and CLaP — classification-based time series segmentation.

Description
-----------

This package contains two related algorithms that share core routines:

**ClaSP** (Classification Score Profile) is a parameter-free method for change
point detection.  It hierarchically splits a time series by training a binary
classifier (k-NN by default) for each possible split point and selecting the
point that best separates the two halves.  The number of change points and
the window size are both learned automatically from the data.  ClaSP is fast,
domain-agnostic, and requires no distributional assumptions.

**CLaP** (Classification Label Profile) extends ClaSP from change point
detection to *state detection*.  Instead of a k-NN classifier, CLaP uses a
time series classifier (e.g. ROCKET) and cross-validates segment-labelled
subsequences.  Segments with high confusion are iteratively merged, producing
a set of state labels that can be reused across non-contiguous time regions.

Both algorithms support three usage modes:

- **Fully unsupervised** — the number of segments is learned automatically.
- **Guided semi-supervised** — provide ``n_segments`` or ``n_change_points``.
  The count is returned as it is, fewer only when the segments are too short
  to split further: ClaSP validates its change points only when it learns
  their number (claspy validates them in both modes).
- **Exact semi-supervised** — provide pre-computed ``change_points``.

| **Type:** change point detection (ClaSP) / state detection (CLaP)
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** ClaSP :math:`O(K\,d\,n^{2})`, CLaP
  :math:`O(K\,n\,(d\,n + K\log K))`, where *K* is the number of change points

ClaSP parameters
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 10 12 56

   * - Name
     - Type
     - Default
     - Description
   * - ``n_segments``
     - str / int
     - ``"learn"``
     - Number of segments (``"learn"`` = auto-detect, each change point
       validated; an integer = guided mode, no validation).
   * - ``n_change_points``
     - int / None
     - ``None``
     - Number of change points (guided mode, no validation; ``0`` = none).
   * - ``n_estimators``
     - int
     - ``10``
     - Number of ClaSPs in the ensemble.
   * - ``window_size``
     - str / int
     - ``"suss"``
     - Window size or auto method (``"suss"``, ``"fft"``, ``"acf"``).
   * - ``k_neighbours``
     - int
     - ``3``
     - Number of nearest neighbours for ClaSP.
   * - ``excl_radius``
     - int
     - ``5``
     - Exclusion radius (multiples of window size).
   * - ``distance``
     - str
     - ``"znormed_euclidean_distance"``
     - Distance function for k-NN.
   * - ``score``
     - str
     - ``"roc_auc"``
     - Scoring function for the profile (``"roc_auc"`` or ``"f1"``).
   * - ``early_stopping``
     - bool
     - ``True``
     - Stop early when no significant split is found.
   * - ``validation``
     - str
     - ``"significance_test"``
     - Validation method for change point significance (learned number of
       segments only).
   * - ``threshold``
     - str / float
     - ``"default"``
     - Threshold for the validation test.
   * - ``n_jobs``
     - int
     - ``-1``
     - Number of parallel jobs.

CLaP parameters
~~~~~~~~~~~~~~~

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

   from tsseg.algorithms import ClaspDetector, ClapDetector

   # --- ClaSP (change point detection, fully unsupervised) ---
   clasp = ClaspDetector()
   cps = clasp.fit_predict(X)                    # change point indices

   # --- CLaP (state detection, guided) ---
   clap = ClapDetector(n_segments=4, classifier="rocket")
   states = clap.fit_predict(X)                  # one state label per time point

**Implementation:** Adapted from original ClaSP code by Arik Ermshaus.  BSD
3-Clause.

**Reference:** Ermshaus, Schäfer & Leser (2023), *ClaSP: parameter-free time
series segmentation*, DMKD; Ermshaus, Schäfer & Leser (2024), *CLaP — State
Detection from Time Series*.

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

.. automodule:: tsseg.algorithms.clap.clasp_detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.clap.clasp_knn
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.distance
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.nearest_neighbour
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.scoring
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.segmentation
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.utils
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.validation
   :show-inheritance:
   :no-index:

.. automodule:: tsseg.algorithms.clap.window_size
   :show-inheritance:
   :no-index:
