tsseg.algorithms.bocd package
=============================

BOCD (Bayesian Online Change-point Detection) — posterior thresholding.

Description
-----------

This offline Bayesian change-point detector integrates out the mean and variance
of each segment under a conjugate Normal-Gamma prior, constructs a run-length
posterior via dynamic programming, and selects change points by thresholding the
posterior probability of a boundary.  It extends the classical framework of
Fearnhead (2006) and Adams & MacKay (2007) to multivariate time series.

A constant hazard function :math:`H(\tau) = 1/\lambda` controls the prior
expectation of segment length.  Two strategies handle multivariate inputs:

- ``"l2"`` — reduces channels via L2 norm before inference.
- ``"ensembling"`` — runs per-channel inference and aggregates change points.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``hazard_lambda``
     - float
     - ``300``
     - Expected run length (constant hazard = ``1/lambda``).
   * - ``mu``
     - float
     - ``0.0``
     - Prior mean of segment observations.
   * - ``kappa``
     - float
     - ``1.0``
     - Strength of the prior mean (normal precision).
   * - ``alpha``
     - float
     - ``1.0``
     - Shape of the inverse-gamma prior over variance.
   * - ``beta``
     - float
     - ``1.0``
     - Scale of the inverse-gamma prior over variance.
   * - ``truncate``
     - int
     - ``-40``
     - Log-probability truncation threshold for the DP matrix.
   * - ``cp_prob_threshold``
     - float
     - ``0.05``
     - Minimum posterior probability to accept a change point.
   * - ``min_distance``
     - int
     - ``25``
     - Minimum distance (samples) between successive change points.
   * - ``max_cps``
     - int / None
     - ``None``
     - Optional upper bound on the number of change points.
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

   from tsseg.algorithms import BOCDDetector

   detector = BOCDDetector(hazard_lambda=200, cp_prob_threshold=0.1)
   labels = detector.fit_predict(X)

**Implementation:** Adapted from hildensia/bayesian_changepoint_detection.
Apache License 2.0.

**Reference:** Fearnhead (2006), Statistics and Computing; Adams & MacKay (2007),
arXiv.

Submodules
----------

tsseg.algorithms.bocd.bayesian\_models module
---------------------------------------------

.. automodule:: tsseg.algorithms.bocd.bayesian_models
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.bocd.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.bocd.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.bocd
   :show-inheritance:
