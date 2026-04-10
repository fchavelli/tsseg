tsseg.algorithms.changefinder package
=====================================

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

This online, single-pass design yields :math:`O(n \cdot k^2)` time complexity,
making ChangeFinder efficient for long time series.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised (``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n \cdot k^2)` where *k* is the AR order

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
     - Minimum score for a peak.  If ``None``, uses ``mean + 2*std`` of the
       score curve.
   * - ``min_distance``
     - int
     - ``10``
     - Minimum samples between successive change points.
   * - ``multivariate_strategy``
     - str
     - ``"l2"``
     - ``"l2"`` (reduce via L2 norm) or ``"ensembling"`` (per-channel).
   * - ``tolerance``
     - float
     - ``0``
     - Tolerance for aggregating change points in ensembling mode.
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

Submodules
----------

tsseg.algorithms.changefinder.sdar module
------------------------------------------

.. automodule:: tsseg.algorithms.changefinder.sdar
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.changefinder.detector module
----------------------------------------------

.. automodule:: tsseg.algorithms.changefinder.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.changefinder
   :show-inheritance:
