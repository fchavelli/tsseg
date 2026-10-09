Seeded Binary Segmentation
==========================

Seeded binary segmentation (SeedBS) and wild binary segmentation (WBS).

Description
-----------

Binary segmentation searches its best split on the whole series, then on each
half: a change point lying between two others of opposite sign can stay
hidden. SeedBS computes the best split of every interval of a fixed,
deterministic collection, the *seeded intervals*: layers of evenly spread,
overlapping intervals whose length decays by a factor ``decay`` from one layer
to the next, :math:`O(n \log n)` points in all. A selection rule then keeps
some of these candidates:

* ``"greedy"``: the largest gain first; every interval containing it is
  dropped, and so on (as in WBS);
* ``"narrowest"``: among the intervals whose gain exceeds the threshold, the
  narrowest first (narrowest over threshold), which the consistency theory of
  the paper covers.

The number of change points comes from ``n_cps``, from a threshold
``penalty`` on the gain, or, by default, from the strengthened Schwarz
information criterion (sSIC) along the solution path, as in the experiments
of the paper. ``intervals="wild"`` draws ``n_intervals`` random intervals
instead (WBS).

The gain of a split is the decrease of a ruptures cost: for ``"l2"``, the
squared CUSUM statistic of the papers, summed over channels.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n \log n)` cost evaluations (seeded intervals)

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
     - Number of change points (guided mode). Takes precedence over
       ``penalty``.
   * - ``penalty``
     - float / None
     - ``None``
     - Threshold on the gain. ``None``: the sSIC picks the model
       (``model="l2"`` only).
   * - ``intervals``
     - str
     - ``"seeded"``
     - ``"seeded"`` (SeedBS) or ``"wild"`` (WBS, random intervals).
   * - ``decay``
     - float
     - :math:`2^{-1/2}`
     - Decay of the seeded interval lengths, in :math:`[1/2, 1)`; closer to
       1: more intervals, slower and more accurate.
   * - ``n_intervals``
     - int
     - ``5000``
     - Number of random intervals (``intervals="wild"``).
   * - ``selection``
     - str
     - ``"greedy"``
     - ``"greedy"`` or ``"narrowest"``.
   * - ``alpha``
     - float
     - ``1.01``
     - Exponent of the sSIC penalty :math:`k\,(d+1)/2\,\log(n)^\alpha`.
   * - ``max_cps``
     - int
     - ``50``
     - Largest number of change points the sSIC considers (the criterion
       needs a bound); a series with more gets ``max_cps`` at most.
   * - ``model``
     - str
     - ``"l2"``
     - Ruptures cost (``"l2"``, ``"l1"``, ``"rbf"``, ``"linear"``,
       ``"normal"``, ``"cosine"``).
   * - ``min_size``
     - int
     - ``1``
     - Minimum number of points on each side of a split.
   * - ``cost_params``
     - dict / None
     - ``None``
     - Keyword arguments of the ruptures cost.
   * - ``backend``
     - str
     - ``"auto"``
     - Backend of the best split, as for :doc:`binseg`.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed of the random intervals.
   * - ``axis``
     - int
     - ``0``
     - Axis representing time.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import SeededBinSegDetector

   # Unknown number of change points: sSIC
   change_points = SeededBinSegDetector().fit_predict(X)

   # Narrowest-over-threshold selection, known number of change points
   detector = SeededBinSegDetector(selection="narrowest", n_cps=3)
   change_points = detector.fit_predict(X)

   # Wild binary segmentation
   detector = SeededBinSegDetector(intervals="wild", random_state=0)
   change_points = detector.fit_predict(X)

**Implementation:** written from the papers (the authors' code has no license;
the R packages ``wbs`` and ``not`` are GPL-2.0-only). The sSIC assumes
Gaussian noise; for a multivariate series each channel has its own variance
and each change point counts :math:`d + 1` parameters, an extension that is
not in the papers. The gain sums the channels as they are: put them on
comparable scales first.

**References:** Kovács, Bühlmann, Li and Munk (2023), Biometrika; Fryzlewicz
(2014), The Annals of Statistics; Baranowski, Chen and Fryzlewicz (2019),
Journal of the Royal Statistical Society B.

API reference
-------------

.. automodule:: tsseg.algorithms.seeded_binseg.detector
   :members:
   :show-inheritance:
   :undoc-members:
