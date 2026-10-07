Evaluation
==========

The :mod:`tsseg.metrics` namespace provides classes implementing common
segmentation quality measures. Every metric inherits from
:class:`tsseg.metrics.BaseMetric` and shares the same ``compute(y_true, y_pred)``
contract: it returns a dictionary with at least a ``"score"`` key (calling the
metric object directly is a shortcut for ``compute``).

Change-point metrics
--------------------

These metrics compare two sets of sparse change-point indices.
``F1Score``, ``Covering``, ``BidirectionalCovering`` and ``GaussianF1Score``
also need the series length: pass it as ``compute(y_true, y_pred,
n_timepoints=n)``, and the change points may then include or omit ``0`` and
``n``. Without it, the length is read from the change points (``y_true`` must
end with it) and a ``UserWarning`` is emitted; with dense labels and
``convert_labels_to_segments=True``, it is the length of the labels.

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

   from tsseg.algorithms import DynpDetector
   from tsseg.algorithms.utils import extract_cps
   from tsseg.data.datasets import load_mocap
   from tsseg.metrics import F1Score, StateMatchingScore

   X, y = load_mocap(trial=0)

   # Change-point evaluation: indices without 0 and n on both sides
   y_pred_cps = DynpDetector(n_cps=len(extract_cps(y))).fit_predict(X)
   f1 = F1Score(margin=0.01)  # a float margin is a fraction of n_timepoints
   result = f1.compute(extract_cps(y), y_pred_cps, n_timepoints=len(y))
   print(result["score"], result["precision"], result["recall"])

   # State-labelling evaluation: one label per time point
   sms = StateMatchingScore()
   result = sms.compute(y, y)
   print(result["score"])  # 1.0

See :doc:`guides/getting-started` for end-to-end examples that combine
detectors and metrics.
