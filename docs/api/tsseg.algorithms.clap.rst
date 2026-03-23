tsseg.algorithms.clap package
=============================

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
- **Exact semi-supervised** — provide pre-computed ``change_points``.

| **Type:** change point detection (ClaSP) / state detection (CLaP)
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate

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
     - Number of segments (``"learn"`` = auto-detect).
   * - ``n_change_points``
     - int / None
     - ``None``
     - Exact number of change points (semi-supervised mode).
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
     - Scoring function for the profile.
   * - ``early_stopping``
     - bool
     - ``True``
     - Stop early when no significant split is found.
   * - ``validation``
     - str
     - ``"significance_test"``
     - Validation method for change point significance.
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
     - Time series classifier (``"rocket"``, ``"knn"``, ``"svm"``).
   * - ``merge_score``
     - str
     - ``"cgain"``
     - Scoring function for segment merging.
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
     - Number of change points (overrides ``n_segments``).
   * - ``change_points``
     - list / None
     - ``None``
     - Pre-computed change points for exact mode.
   * - ``n_segments``
     - int / None
     - ``None``
     - Number of segments.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ClaspDetector, ClapDetector

   # --- ClaSP (change point detection, fully unsupervised) ---
   clasp = ClaspDetector()
   cps = clasp.fit_predict(X)                    # dense labels

   # --- CLaP (state detection, guided) ---
   clap = ClapDetector(n_segments=4, classifier="rocket")
   states = clap.fit_predict(X)                  # sparse state labels

**Implementation:** Adapted from original ClaSP code by Arik Ermshaus.  BSD
3-Clause.

**Reference:** Ermshaus, Schäfer & Leser (2023), *ClaSP: parameter-free time
series segmentation*, DMKD; Ermshaus, Schäfer & Leser (2024), *CLaP — State
Detection from Time Series*.

Submodules
----------

tsseg.algorithms.clap.clap module
---------------------------------

.. automodule:: tsseg.algorithms.clap.clap
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.clap\_detector module
-------------------------------------------

.. automodule:: tsseg.algorithms.clap.clap_detector
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.clasp\_detector module
--------------------------------------------

.. automodule:: tsseg.algorithms.clap.clasp_detector
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.clasp\_knn module
---------------------------------------

.. automodule:: tsseg.algorithms.clap.clasp_knn
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.distance module
-------------------------------------

.. automodule:: tsseg.algorithms.clap.distance
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.nearest\_neighbour module
-----------------------------------------------

.. automodule:: tsseg.algorithms.clap.nearest_neighbour
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.scoring module
------------------------------------

.. automodule:: tsseg.algorithms.clap.scoring
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.segmentation module
-----------------------------------------

.. automodule:: tsseg.algorithms.clap.segmentation
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.utils module
----------------------------------

.. automodule:: tsseg.algorithms.clap.utils
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.validation module
---------------------------------------

.. automodule:: tsseg.algorithms.clap.validation
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.clap.window\_size module
-----------------------------------------

.. automodule:: tsseg.algorithms.clap.window_size
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.clap
   :members:
   :show-inheritance:
   :undoc-members:
