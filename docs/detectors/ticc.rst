TICC
====

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

Without ``n_states``, the number of states is selected by the Bayesian
information criterion (BIC), as the paper recommends: TICC is fitted for each
number of states from 1 to ``max_states``, and the fit with the lowest BIC is
kept (see :class:`~tsseg.algorithms.ticc.detector.TiccDetector`).

| **Type:** state detection
| **Supervision:** unsupervised (``n_states=None``, BIC) or semi-supervised
  (``n_states`` given)
| **Scope:** multivariate (transductive)
| **Complexity:** :math:`O(S\,(n\,(d\,w)^{2} + (d\,w)^{3}))` per iteration,
  where *S* is ``n_states`` and *w* is ``window_size``: each stacked window
  (dimension :math:`d\,w`) is scored under the precision matrix of each state.
  Without ``n_states``, one fit per number of states up to ``max_states``.

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
     - int / None
     - ``None``
     - Number of states (clusters); ``None`` selects it by BIC.
   * - ``max_states``
     - int
     - ``8``
     - Largest number of states tried when ``n_states`` is ``None``.
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

   # number of states chosen by BIC
   detector = TiccDetector(window_size=10, beta=200, max_states=6)
   states = detector.fit_predict(X)
   detector.n_states_, detector.bic_

**Implementation:** Adapted from the original TICC code by Hallac et al.

**Reference:** Hallac, Vare, Boyd & Leskovec (2017), *Toeplitz Inverse
Covariance-Based Clustering of Multivariate Time Series Data*, KDD.

API reference
-------------

.. automodule:: tsseg.algorithms.ticc.detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.ticc.ticc
   :members:
   :show-inheritance:
   :undoc-members:
