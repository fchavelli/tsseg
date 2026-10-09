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
3/4 quantiles of the segment and maximises the resulting approximate gain
curves, then refits the classifier at the best of these guesses and maximises
the new gain curve over all candidate splits, so that a handful of classifier
fits per segment suffice. A pseudo-permutation test of the first-step gains
decides whether the best split is kept, and binary segmentation recurses on
the kept splits.

Three gains are available: the random forest classifier (the paper's method),
a k-nearest-neighbour classifier (``"knn"``) and the parametric change in mean
(``"change_in_mean"``). The search can use binary (the paper's method), seeded
binary or wild binary segmentation.

With ``n_cps=K`` (semi-supervised mode, not part of the paper), the detector
runs binary segmentation best-first: it repeatedly splits the segment whose
best split has the largest gain, as computed by the package, until ``K``
change points are found; the significance test is not used.

| **Type:** change point detection
| **Supervision:** unsupervised (permutation test) or semi-supervised (``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** random forest: a few forests per segment, each nearly linear
  in :math:`n` for a fixed tree depth (Section 3.3 of the paper); empirically
  close to linear in :math:`n` (Section 4.5). ``"knn"``: quadratic in time and
  memory (an :math:`n \times n` distance matrix and its ordering, about
  :math:`16 n^2` bytes)
| **Requires:** changeforest (``pip install tsseg[changeforest]``)

Limitations
-----------

* **Independent observations.** The method assumes them (Section 2 of the
  paper) and over-segments autocorrelated series: three stationary AR(1)
  series with coefficient 0.9 (2000 points, 3 channels, no change) give 63 to
  71 change points each. Section 5 of the paper suggests adding lagged
  observations as extra channels.
* **Scale.** Only the random forest gain is invariant to the scale of each
  channel. ``"knn"`` uses Euclidean distances, and the ``"change_in_mean"``
  gain and its threshold :math:`\log(n)(d + 1)` assume noise of unit variance
  (i.i.d. Gaussian noise of standard deviation 3 gives about 50 change points
  in 600 points). Standardise the channels first, for instance by the median
  absolute deviation of their consecutive differences (Section 4.2).
* **Seeded and wild binary segmentation.** The model selection is designed
  for binary segmentation; the paper does not evaluate the other two and notes
  that they would need a different model selection to avoid overfitting
  (Section 5). On 20 series of i.i.d. noise (600 points, 5 channels), ``"bs"``
  returned no change point, ``"sbs"`` and ``"wbs"`` at least one on 11 and 9
  of them.
* **Guided mode.** Segments are ranked by their raw gain, which is negative on
  a homogeneous segment and decreases with its length: change points beyond
  the true ones tend to fall in the shortest segments.

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
     - Threshold on the p-value of the pseudo-permutation test, a tuning
       parameter rather than a valid significance level (Section 3.4).
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
     - Decay of the seeded intervals (``"sbs"``), in :math:`(0, 1)`.
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
     - Features tried per split: ``"sqrt"`` (:math:`\lfloor\sqrt{d}\rfloor`),
       ``None`` (all) or an integer.
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
