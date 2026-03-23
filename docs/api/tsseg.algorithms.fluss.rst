tsseg.algorithms.fluss package
==============================

FLUSS — Fast Low-cost Unipotent Semantic Segmentation.

Description
-----------

FLUSS is a domain-agnostic method for semantic segmentation of time series.  It
leverages the *matrix profile* (computed by ``stumpy``) and derives the *arc
curve*: for each time index the number of nearest-neighbour arcs crossing that
index is counted.  Valleys in the arc curve indicate semantic boundaries because
few matching subsequences span across a true regime change.

A sequential peak search on the inverted arc curve yields the requested number
of change points.  FLUSS has only one parameter — the window size — which
corresponds to the dominant period length.

| **Type:** change point detection
| **Supervision:** semi-supervised or unsupervised
| **Scope:** univariate (multivariate via ensembling)
| **Requires:** ``stumpy``

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 58

   * - Name
     - Type
     - Default
     - Description
   * - ``window_size``
     - int
     - ``10``
     - Sliding window size (use the dominant period length).
   * - ``n_segments``
     - int
     - ``2``
     - Number of segments (= change points + 1).
   * - ``exclusion_factor``
     - int
     - ``5``
     - Multiplying factor for the exclusion zone around detected points.
   * - ``multivariate_strategy``
     - str
     - ``"ensembling"``
     - Strategy for multivariate inputs (``"ensembling"``).
   * - ``tolerance``
     - float
     - ``0.01``
     - Tolerance for aggregating change points.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import FLUSSDetector

   detector = FLUSSDetector(window_size=50, n_segments=4)
   labels = detector.fit_predict(X)

**Implementation:** Wrapper around ``stumpy``.  BSD 3-Clause.

**Reference:** Gharghabi, Yeh, Ding, Ding, Hibbing, LaMunion, Kaplan, Crouter &
Keogh (2017, 2019), *Domain Agnostic Online Semantic Segmentation*, ICDM / DMKD.

Submodules
----------

tsseg.algorithms.fluss.detector module
--------------------------------------

.. automodule:: tsseg.algorithms.fluss.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.fluss
   :members:
   :show-inheritance:
   :undoc-members:
