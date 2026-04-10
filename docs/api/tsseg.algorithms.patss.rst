tsseg.algorithms.patss package
==============================

PaTSS — Pattern-based Time Series Segmentation.

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

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   tsseg.algorithms.patss.algorithms
   tsseg.algorithms.patss.embedding
   tsseg.algorithms.patss.segmentation

Submodules
----------

tsseg.algorithms.patss.detector module
--------------------------------------

.. automodule:: tsseg.algorithms.patss.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.patss
   :show-inheritance:
