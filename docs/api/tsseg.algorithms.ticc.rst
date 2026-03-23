tsseg.algorithms.ticc package
=============================

TICC — Toeplitz Inverse Covariance-based Clustering.

Description
-----------

TICC simultaneously segments and clusters multivariate time series by modelling
each cluster as a Markov Random Field (MRF) defined over a short window of
length ``window_size``.  The MRF is characterised by a block-Toeplitz inverse
covariance matrix that captures temporal and cross-variable partial correlations.

The algorithm alternates between:

1. **Assignment step** — dynamic programming assigns each time step to the
   cluster maximising a log-likelihood + switching penalty :math:`\beta`.
2. **Update step** — ADMM updates the Toeplitz inverse covariance of each
   cluster subject to a sparsity penalty :math:`\lambda`.

| **Type:** state detection
| **Supervision:** semi-supervised (``n_states`` required)
| **Scope:** multivariate (transductive)

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
     - Sliding window size.
   * - ``n_states``
     - int
     - ``5``
     - Number of states (clusters).
   * - ``lambda_parameter``
     - float
     - ``0.11``
     - Sparsity parameter for the inverse covariance.
   * - ``beta``
     - float
     - ``400``
     - Switching penalty (temporal consistency).
   * - ``maxIters``
     - int
     - ``100``
     - Maximum EM iterations.
   * - ``threshold``
     - float
     - ``2e-5``
     - Convergence threshold.
   * - ``num_proc``
     - int
     - ``1``
     - Number of parallel processes.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import TiccDetector

   detector = TiccDetector(n_states=4, window_size=10, beta=200)
   states = detector.fit_predict(X)

**Implementation:** Adapted from the original TICC code by Hallac et al.

**Reference:** Hallac, Vare, Boyd & Leskovec (2017), *Toeplitz Inverse
Covariance-Based Clustering of Multivariate Time Series Data*, KDD.

Submodules
----------

tsseg.algorithms.ticc.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.ticc.detector
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.ticc.ticc module
---------------------------------

.. automodule:: tsseg.algorithms.ticc.ticc
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.ticc
   :members:
   :show-inheritance:
   :undoc-members:
