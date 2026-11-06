Evaluation
==========

The :mod:`tsseg.metrics` namespace provides classes implementing common
segmentation quality measures. Every metric inherits from
:class:`~tsseg.metrics.BaseMetric` and shares the same ``compute(y_true, y_pred)``
contract: it returns a dictionary with at least a ``"score"`` key (calling the
metric object directly is a shortcut for ``compute``).

Change point metrics
--------------------

These metrics compare two sets of sparse change point indices.
``F1Score``, ``Covering``, ``BidirectionalCovering`` and ``GaussianF1Score``
also need the series length: pass it as ``compute(y_true, y_pred,
n_timepoints=n)``, and the change points may then include or omit ``0`` and
``n``. Without it, the length is read from the change points (``y_true`` must
end with it) and a ``UserWarning`` is emitted; with dense labels and
``convert_labels_to_segments=True``, it is the length of the labels.
``HausdorffDistance`` takes the same ``n_timepoints`` and
``convert_labels_to_segments`` arguments: it then compares only the change
points inside the series, without ``0`` and ``n``. Without either, it
compares the two lists as they are given, boundaries included.

The figures below score one prediction, on a series of 1,000 points with true
change points at 300, 600 and 800 and predicted ones at 320, 450 and 790. The
scores they show are computed by the metrics when the documentation is built.

F1 score
~~~~~~~~

``F1Score`` matches each true change point with the closest predicted one
within ``margin`` points (a float is a fraction of the series length), each
prediction being matched at most once. Precision is the share of matched
predictions, recall the share of matched true change points.

.. tsseg-metric-figure:: f1

Gaussian F1 score
~~~~~~~~~~~~~~~~~

``GaussianF1Score`` replaces the window by a Gaussian reward of width
:math:`\sigma` = ``sigma_fraction`` · n around each true change point (1 % of
the length by default; 2.5 % in the figure). A prediction earns the height of
the curve at its position, so a close miss keeps partial credit. Precision and
recall divide the total reward by the number of predictions and of true change
points.

.. tsseg-metric-figure:: gaussian_f1

Hausdorff distance
~~~~~~~~~~~~~~~~~~

``HausdorffDistance`` is the largest distance, in points, from a change point
of either set to the nearest change point of the other set (lower is better).
A single missed or spurious change point is enough to make it large.

.. tsseg-metric-figure:: hausdorff

Covering
~~~~~~~~

``Covering`` scores segments rather than change points: each true segment takes
its best intersection over union (IoU) with a predicted segment, and the
average is weighted by the length of the true segments.
``BidirectionalCovering`` also covers the predicted segments by the true ones
and combines the two directions (harmonic mean by default).

.. tsseg-metric-figure:: covering

.. currentmodule:: tsseg.metrics

.. autosummary::
   :toctree: _autogen
   :nosignatures:

   F1Score
   Covering
   HausdorffDistance
   GaussianF1Score
   BidirectionalCovering

State detection metrics
-----------------------

These metrics compare two dense label arrays of equal length. The label values
carry no meaning: predicted states are compared with the true ones up to a
renaming (by pairs of points, or after Hungarian matching for SMS). The figures
use a series of 1,000 points in states A, B, A and C.

ARI, AMI and NMI
~~~~~~~~~~~~~~~~

The clustering scores of scikit-learn: the Adjusted Rand Index counts the pairs
of time points that both labelings put in the same state or in different
states, adjusted for chance; the Normalized and Adjusted Mutual Information
measure how much one labeling tells about the other, AMI being adjusted for
chance. They ignore time: every point counts the same, wherever it is.

.. tsseg-metric-figure:: permutation

WARI and WNMI
~~~~~~~~~~~~~

``WeightedAdjustedRandIndex`` and ``WeightedNormalizedMutualInformation``
weight each time point by :math:`1 + \alpha d`, where *d* is its distance to
the nearest true boundary and :math:`\alpha` = 0.1 by default. A boundary
placed a little late costs less than the same number of wrong points inside a
segment.

.. tsseg-metric-figure:: weighted

State Matching Score
~~~~~~~~~~~~~~~~~~~~

``StateMatchingScore`` (SMS) first maps the predicted states to the true ones
with the Hungarian algorithm. Each run of wrong points with the same predicted
label is an error segment, typed by what it covers: a **delay** continues the
neighbouring state (a boundary placed late or early), an **isolation** is a
spurious state inside one true segment, a **transition** covers one true
boundary and a **missing** error covers several, so that a whole true segment
is lost. The score is :math:`1 - (L + P) / n`, with *L* the number of wrong
points and *P* their penalties, weighted by type (``weights``); isolations and
transitions are also weighted by their distance to the nearest true boundary.

.. tsseg-metric-figure:: sms

.. autosummary::
   :toctree: _autogen
   :nosignatures:

   AdjustedRandIndex
   AdjustedMutualInformation
   NormalizedMutualInformation
   WeightedAdjustedRandIndex
   WeightedNormalizedMutualInformation
   StateMatchingScore

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

   # Change point evaluation: indices without 0 and n on both sides
   y_pred_cps = DynpDetector(n_cps=len(extract_cps(y))).fit_predict(X)
   f1 = F1Score(margin=0.01)  # a float margin is a fraction of n_timepoints
   result = f1.compute(extract_cps(y), y_pred_cps, n_timepoints=len(y))
   print(result["score"], result["precision"], result["recall"])

   # State detection evaluation: one label per time point
   sms = StateMatchingScore()
   result = sms.compute(y, y)
   print(result["score"])  # 1.0

See :doc:`guides/getting-started` for end-to-end examples that combine
detectors and metrics.
