Evaluation
==========

The :mod:`tsseg.metrics` namespace provides classes implementing common
segmentation quality measures. Every metric inherits from
:class:`tsseg.metrics.BaseMetric` and shares the same ``compute(y_true, y_pred)``
contract (with an ``update`` / ``compute`` variant for streaming use).

Change-point metrics
--------------------

These metrics compare two sets of sparse change-point indices.

.. autosummary::
   :toctree: _autogen
   :nosignatures:

   tsseg.metrics.F1Score
   tsseg.metrics.Covering
   tsseg.metrics.HausdorffDistance
   tsseg.metrics.GaussianF1Score
   tsseg.metrics.BidirectionalCovering

State-labelling metrics
-----------------------

These metrics compare two dense label arrays of equal length, aligning
predicted to ground-truth states when needed (e.g. Hungarian matching).

.. autosummary::
   :toctree: _autogen
   :nosignatures:

   tsseg.metrics.AdjustedRandIndex
   tsseg.metrics.AdjustedMutualInformation
   tsseg.metrics.NormalizedMutualInformation
   tsseg.metrics.WeightedAdjustedRandIndex
   tsseg.metrics.WeightedNormalizedMutualInformation
   tsseg.metrics.StateMatchingScore

Base protocol
-------------

.. autoclass:: tsseg.metrics.BaseMetric
   :members:
   :show-inheritance:

Usage
-----

.. code-block:: python

   from tsseg.metrics import F1Score, StateMatchingScore

   # Change-point evaluation
   f1 = F1Score(margin=5)
   result = f1.compute(y_true_cps, y_pred_cps)
   print(result["f1"])

   # State-labelling evaluation
   sms = StateMatchingScore()
   result = sms.compute(y_true_labels, y_pred_labels)
   print(result["score"])

See :doc:`guides/getting-started` for end-to-end examples that combine
detectors and metrics.
