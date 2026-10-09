changeforest
============

Classifier-based nonparametric change point detection with random forests.

Description
-----------

changeforest (Londschien, Bühlmann and Kovács, 2023) detects changes in the
joint distribution of a multivariate series without assuming a parametric
model inside the segments. To score a split of a segment, it trains a
classifier to separate the observations before the split from those after it,
and turns the out-of-bag class probabilities into a classifier log-likelihood
ratio, the *gain*. A two-step search fits the classifier at the 1/4, 1/2 and
3/4 quantiles of the segment, maximises the resulting approximate gain curves
and refines around the best guess, so that a handful of classifier fits per
segment suffice. A permutation test decides whether the best split is
significant, and binary segmentation recurses on the significant splits.

Three gains are available: the random forest classifier (the paper's method),
a k-nearest-neighbour classifier (``"knn"``) and the parametric change in mean
(``"change_in_mean"``). The search can use binary, seeded binary or wild
binary segmentation.

With ``n_cps=K`` (semi-supervised mode, not part of the paper), the detector
runs binary segmentation best-first: it repeatedly splits the segment whose
best split has the largest gain, as computed by the package, until ``K``
change points are found; the significance test is not used.

| **Type:** change point detection
| **Supervision:** unsupervised (permutation test) or semi-supervised (``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** a few forests per segment, each nearly linear in :math:`n`
  for a fixed tree depth (Section 3.3 of the paper); empirically close to
  linear in :math:`n` (Section 4.5)
| **Requires:** changeforest (``pip install tsseg[changeforest]``)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``method``
     - str
     - ``"random_forest"``
     - Gain: ``"random_forest"``, ``"knn"`` or ``"change_in_mean"``.
   * - ``segmentation``
     - str
     - ``"bs"``
     - Binary (``"bs"``), seeded binary (``"sbs"``) or wild binary
       (``"wbs"``) segmentation.
   * - ``minimal_relative_segment_length``
     - float
     - ``0.01``
     - Minimal segment length as a fraction :math:`\delta` of the series
       length (segments have at least :math:`\lceil \delta n \rceil` points).
   * - ``model_selection_alpha``
     - float
     - ``0.02``
     - Significance level of the permutation test.
   * - ``model_selection_n_permutations``
     - int
     - ``199``
     - Number of permutations of the test.
   * - ``minimal_gain_to_split``
     - float / None
     - ``None``
     - Minimal gain to split with ``"change_in_mean"``; ``None`` uses
       :math:`\log(n)(d + 1)`.
   * - ``number_of_wild_segments``
     - int
     - ``100``
     - Number of random intervals (``"wbs"``).
   * - ``seeded_segments_alpha``
     - float
     - :math:`1/\sqrt{2}`
     - Decay of the seeded intervals (``"sbs"``), in :math:`[1/2, 1)`.
   * - ``random_forest_n_estimators``
     - int
     - ``100``
     - Trees per forest.
   * - ``random_forest_max_depth``
     - int / None
     - ``8``
     - Maximal tree depth (``None``: unlimited).
   * - ``random_forest_max_features``
     - int / str / None
     - ``"sqrt"``
     - Features tried per split: ``"sqrt"``, ``None`` (all) or an integer.
   * - ``random_forest_n_jobs``
     - int
     - ``-1``
     - Threads used to grow the forests (``-1``: all cores).
   * - ``random_state``
     - int
     - ``0``
     - Seed of the forests, the permutations and the random intervals.
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points (semi-supervised mode).

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ChangeForestDetector

   # Unsupervised: the permutation test selects the change points
   change_points = ChangeForestDetector().fit_predict(X)

   # Semi-supervised: exactly three change points
   change_points = ChangeForestDetector(n_cps=3).fit_predict(X)

**Implementation:** wraps the official
`changeforest <https://github.com/mlondschien/changeforest>`_ package (Rust,
BSD-3-Clause, ``pip install tsseg[changeforest]``); defaults are those of the
package, which match the paper's main simulations.

**Reference:** Londschien, Bühlmann and Kovács (2023), *Random Forests for
Change Point Detection*, Journal of Machine Learning Research 24(216).
