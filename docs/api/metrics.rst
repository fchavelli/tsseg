.. _api-metrics:

Metric helpers
==============

The :mod:`tsseg.metrics` namespace provides classes implementing common
segmentation quality measures. Metrics share a ``__call__`` signature accepting
predictions and ground-truth annotations, and many expose extra helpers for
per-segment analysis.

Change-point metrics
--------------------

* :class:`tsseg.metrics.F1Score` – balanced F1 score computed from matched
  change points with tolerance windows.
* :class:`tsseg.metrics.Covering` – overlap-based covering score between true
  and predicted segments.
* :class:`tsseg.metrics.HausdorffDistance` – worst-case deviation between true
  and predicted change points.
* :class:`tsseg.metrics.GaussianF1` – F1 score using Gaussian-weighted
  tolerance around true change points.
* :class:`tsseg.metrics.BidirectionalCovering` – symmetric segment-overlap
  score measuring both precision and recall coverage.

State-labelling metrics
-----------------------

* :class:`tsseg.metrics.AdjustedRandIndex` – adjusts the Rand index for chance
  agreement.
* :class:`tsseg.metrics.AdjustedMutualInformation` – normalized mutual
  information corrected for chance.
* :class:`tsseg.metrics.NormalizedMutualInformation` – mutual information
  normalized by joint entropy.
* :class:`tsseg.metrics.WeightedAdjustedRandIndex` – class-imbalance aware
  variant of the adjusted Rand index.
* :class:`tsseg.metrics.WeightedNormalizedMutualInformation` – weighted version
  of the normalized mutual information.
* :class:`tsseg.metrics.StateMatchingScore` – Hungarian matching-based score
  that first aligns predicted and true states before computing accuracy.

Base protocol
-------------

All metrics inherit from :class:`tsseg.metrics.BaseMetric`, which defines the
common ``update``/``compute`` contract used by collection utilities. Refer to
:doc:`../guides/getting-started` for integration examples.
