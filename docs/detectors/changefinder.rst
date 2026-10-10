ChangeFinder
============

ChangeFinder — Online two-stage change-point detection via outlier scoring.

Description
-----------

ChangeFinder reduces change-point detection to outlier detection using a pair of
sequentially discounting autoregressive (SDAR) models.  The algorithm operates in
two stages:

1. **Stage 1:** An SDAR model is fitted online on the raw signal.  Each
   observation receives an outlier score (logarithmic or quadratic loss), which
   is smoothed by a causal moving average of width *T*.

2. **Stage 2:** A second SDAR model is fitted on the smoothed scores from
   Stage 1.  A second moving average produces the final *change-point score*
   curve.  Peaks in this curve are returned as change points.

The SDAR model maintains an AR(\ *k*\ ) model with exponential discounting:
older observations are progressively down-weighted at rate *r*, allowing the
model to track non-stationary dynamics.  AR parameters are updated at each step
by re-solving a discounted Toeplitz system.

This online, single-pass design solves one :math:`k \times k` system per sample and
stage, hence :math:`O(n \cdot k^3)` time after the reduction of the channels,
making ChangeFinder efficient for long time series.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised (``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n + n\,k^{3})` where *k* is the AR order
  (``"l2"`` strategy; ``"ensembling"``: :math:`O(d\,n\,k^{3})`)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``order``
     - int
     - ``5``
     - AR order *k* for both SDAR stages.
   * - ``discount``
     - float
     - ``0.005``
     - Discounting rate *r* in (0, 1).  Smaller values retain longer memory.
   * - ``smooth_window``
     - int
     - ``7``
     - Moving-average window length *T* after each SDAR stage.
   * - ``score``
     - str
     - ``"logarithmic"``
     - Scoring function: ``"logarithmic"`` (neg. log-likelihood) or
       ``"quadratic"`` (squared prediction error).
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points to return.  If ``None``, all peaks above
       threshold are returned.
   * - ``threshold``
     - float / None
     - ``None``
     - Minimum score for a peak.  If ``None``, uses
       ``mean + threshold_factor * std`` of the non-zero scores.
   * - ``threshold_factor``
     - float
     - ``2.0``
     - Standard deviations above the mean of the data-driven threshold.
   * - ``min_distance``
     - int
     - ``10``
     - Minimum samples between successive change points.
   * - ``min_distance_fraction``
     - float
     - ``0.0``
     - Minimum distance between change points as a fraction of the series
       length (the larger of the two minimum distances applies).
   * - ``multivariate_strategy``
     - str
     - ``"l2"``
     - ``"l2"`` (reduce via L2 norm) or ``"ensembling"`` (per-channel).
   * - ``tolerance``
     - float
     - ``0``
     - Tolerance for aggregating change points in ensembling mode (a float
       below 1 is a fraction of the length; without ``n_cps``, 0 means the
       minimum distance).
   * - ``consensus``
     - float
     - ``0.0``
     - Ensembling without ``n_cps``: detections of the channels within
       ``tolerance`` are merged; a merged change point is returned if at
       least this fraction of the channels (one at least) detected it.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ChangeFinderDetector

   detector = ChangeFinderDetector(order=5, discount=0.005, smooth_window=7)
   cps = detector.fit_predict(X)

**Reference:** Takeuchi, J. and Yamanishi, K. (2006), *A unifying framework for
detecting outliers and change points from time series*, IEEE TKDE, vol. 18,
no. 4, pp. 482–492.

API reference
-------------

.. automodule:: tsseg.algorithms.changefinder.sdar
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.changefinder.detector
   :members:
   :show-inheritance:
   :undoc-members:
