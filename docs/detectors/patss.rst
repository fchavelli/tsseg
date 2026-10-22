PaTSS
=====

PaTSS — Pattern-based Time Series Segmentation.

.. warning::
   **Deprecated.** ``PatssDetector`` will be removed in a future release. It
   needs ``npbad``, which is not on PyPI, and it over-segments: on a 54-series
   sample of TSB-SEG it predicted 17 times more change points than the ground
   truth (median), and scored below a random segmentation on univariate series.
   The authors maintain `PaTSEmb <https://github.com/ML-KULeuven/PaTSEmb>`_
   (``pip install patsemb``), which needs a Java runtime.

Description
-----------

PaTSS detects states in time series by mining frequent temporal patterns,
building a pattern-based embedding matrix, and clustering the resulting features.
The pipeline consists of three stages:

1. **Pattern mining** — extracts frequent sub-patterns from the series using
   configurable length, frequency and overlap parameters.
2. **Embedding** — constructs a binary or frequency-based embedding matrix where
   each row is a time step and each column a mined pattern.
3. **Segmentation** — applies a segmentation algorithm (default: logistic
   regression classifier) on the embedding to partition the series into states.

All configuration is passed through a ``config`` dictionary.

| **Type:** state detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,R\,n^{2})` asymptotically: for each channel and
  each of the :math:`R \le 7` window lengths (16 to 1024 by default),
  ``npbad`` cuts the :math:`n` windows with a mask over the whole series. Add
  the Java pattern miner, whose cost depends on the patterns found, then
  k-means and a silhouette score (quadratic in its sample of at most
  :math:`2000 + 0.1\,(n - 2000)` points) for each of the 8 candidate numbers of
  states. The per-window overhead dominates in practice: about 15 s for 4,000
  univariate points, close to linear in :math:`n` up to 8,000 points

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``config``
     - dict / None
     - ``None``
     - Settings for PaTSS.  If ``None``, default hyperparameters are used.
       Keys include ``"pattern_length"``, ``"min_frequency"``,
       ``"max_overlap"``, ``"n_segments"`` and classifier settings.
   * - ``axis``
     - int
     - ``0``
     - Time axis (input assumed ``(n_timepoints, n_channels)``).

Usage
-----

.. code-block:: python

   from tsseg.algorithms import PatssDetector

   detector = PatssDetector()       # default config
   states = detector.fit_predict(X)

   # Custom config
   detector = PatssDetector(config={"pattern_length": 20, "n_segments": 5})
   states = detector.fit_predict(X)

**Implementation:** Adapted from the original PaTSS code.

**Reference:** —

API reference
-------------

.. automodule:: tsseg.algorithms.patss.detector
   :members:
   :show-inheritance:
   :undoc-members:
