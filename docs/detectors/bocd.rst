BOCD
====

BOCD – offline Bayesian change-point detection with posterior thresholding.

Description
-----------

This offline Bayesian change-point detector integrates out the mean and variance
of each segment under a conjugate Normal-Gamma prior and uses forward and backward
messages to compute the marginal probability of a boundary at each position,
following the product-partition model of Fearnhead (2006). It uses the whole
series; the historical class name does not imply the online filtering algorithm
of Adams & MacKay (2007).

A constant hazard function :math:`H(\tau) = 1/\lambda` controls the prior
expectation of segment length. The duration mass is geometric,
:math:`g(l) = H(1-H)^{l-1}`, and the final segment is right censored.
Boundary probabilities are separate marginal events: their sum is the posterior
expected number of changes, and can exceed one. Two strategies handle
multivariate inputs:

- ``"l2"`` – reduces channels via L2 norm before inference.
- ``"ensembling"`` – runs per-channel inference and aggregates change points.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n + n^{2})` worst-case time, :math:`O(n)` working memory
  beyond the input (``"l2"`` strategy; ``"ensembling"``: :math:`O(d\,n^{2})`
  time and up to :math:`O(d\,n)` candidate storage).

The implementation updates segment moments with Welford's recurrence and
precomputes the likelihood terms that depend only on segment length. It does not
store every segment or compute the distribution of the ordinal number of each
change point. Truncation approximates message sums; it does not make this
implementation linear in time, since all segment moments are still evaluated.

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
     - Natural-log cutoff: stop a sum after a term smaller than
       ``exp(truncate)`` times its accumulated total.
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
     - Optional upper bound after thresholding, not inference conditional on
       exactly this many changes.
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
   change_points = detector.fit_predict(X)

**Implementation:** Adapted from hildensia/bayesian_changepoint_detection.
MIT License.

**Reference:** `Fearnhead (2006), Statistics and Computing
<https://eprints.lancs.ac.uk/id/eprint/8189/>`_; `Adams & MacKay (2007), arXiv
<https://arxiv.org/abs/0710.3742>`_.

Suitability and limitations
---------------------------

The observation model is an independent Gaussian distribution within each
segment, with an unknown mean and variance. It suits abrupt changes of level or
variance. A change in autocorrelation, frequency or cross-channel dependence can
leave these statistics unchanged. Temporal dependence and outliers can also
produce extra detections. The conjugate prior must match the scale and offset of
the observations; ``hazard_lambda`` is in samples, so its interpretation depends
on the sampling rate.

L2 reduction discards direction: a multivariate mean changing from ``(a, 0)`` to
``(-a, 0)`` has the same norm. Ensembling preserves changes visible in individual
channels, at a cost proportional to their number; it still does not model their
joint covariance. ``min_distance`` excludes candidates near both series edges
and spaces the retained boundaries. It does not constrain segment lengths in the
Bayesian model.

Before the correction in Unreleased, the detector passed a constant mass instead
of a geometric duration mass, normalised boundary marginals to sum to one, and
used cubic-time inference with quadratic memory. Consequently, earlier benchmark
scores measure that implementation and its decision rule. They need to be
recomputed before drawing conclusions about the corrected model. Performance
improvements alone do not establish detection quality on real data.

API reference
-------------

.. automodule:: tsseg.algorithms.bocd.bayesian_models
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.bocd.detector
   :members:
   :show-inheritance:
   :undoc-members:
