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

1. **Assignment step** — dynamic programming assigns each window to a
   cluster, minimising the negative log-likelihood of the windows plus a
   switching penalty :math:`\beta` for each change of cluster.
2. **Update step** — ADMM updates the Toeplitz inverse covariance of each
   cluster subject to a sparsity penalty :math:`\lambda`.

The label of time point *t* is that of the window of points *t* to
*t* + ``window_size`` − 1. The last ``window_size - 1`` observations enter no
window, and the labels of the last ``2 * (window_size - 1)`` points repeat an
earlier one. ``predict`` on a series other than the one given to ``fit``
labels it with a new fit, with a warning (see the Notes of
:class:`~tsseg.algorithms.ticc.detector.TiccDetector`).

Without ``n_states``, the number of states is selected by the Bayesian
information criterion (BIC), as the paper recommends: TICC is fitted for each
number of states from 1 to ``max_states``, and the fit with the lowest BIC is
kept (see :class:`~tsseg.algorithms.ticc.detector.TiccDetector`).

| **Type:** state detection
| **Supervision:** unsupervised (``n_states=None``, BIC) or semi-supervised
  (``n_states`` given)
| **Scope:** univariate and multivariate
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
     - Number of consecutive time points stacked into each window.
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
     - Weight of the :math:`\ell_1` penalty on the inverse covariances.
   * - ``beta``
     - float
     - ``400``
     - Switching penalty (temporal consistency); twice the :math:`\beta` of
       the paper.
   * - ``maxIters``
     - int
     - ``100``
     - Maximum number of iterations (update and assignment steps).
   * - ``threshold``
     - float
     - ``2e-5``
     - No effect: the solver never reads it.
   * - ``num_proc``
     - int
     - ``1``
     - Number of processes for the update steps; the output does not
       depend on it.
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
