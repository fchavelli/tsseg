ClaSP
=====

ClaSP (Classification Score Profile) – classification-based change point
detection.

Description
-----------

ClaSP is a parameter-free method for change point detection. It hierarchically
splits a time series by training a binary classifier (k-NN by default) for
each possible split point and selecting the point that best separates the two
halves. The number of change points and the window size are both learned
automatically from the data. ClaSP is fast, domain-agnostic, and requires no
distributional assumptions.

ClaSP runs in two modes:

- **Unsupervised** (``n_segments="learn"``, the default): the number of
  segments is learned, and each change point must pass the ``validation``
  test.
- **Guided**: provide ``n_segments`` or ``n_change_points``. The count is
  returned as it is, fewer only when the segments are too short to split
  further; the change points are then not validated (claspy validates them in
  both modes).

The state detector :doc:`CLaP <clap>` starts from the change points of ClaSP
and shares its core routines.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(K\,d\,n^{2})`, where *K* is the number of change
  points

Parameters
----------

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

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ClaspDetector

   # Unsupervised: the number of change points is learned
   cps = ClaspDetector().fit_predict(X)  # change point indices

   # Guided: three change points
   cps = ClaspDetector(n_change_points=3).fit_predict(X)

**Implementation:** Adapted from original ClaSP code by Arik Ermshaus.  BSD
3-Clause.

**Reference:** Ermshaus, Schäfer & Leser (2023), *ClaSP: parameter-free time
series segmentation*, DMKD.

API reference
-------------

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
