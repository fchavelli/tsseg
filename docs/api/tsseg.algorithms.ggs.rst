tsseg.algorithms.ggs package
============================

GGS — Greedy Gaussian Segmentation.

Description
-----------

GGS segments a multivariate time series by modelling each segment as i.i.d.
samples from a multivariate Gaussian distribution.  The algorithm maximises the
total log-likelihood minus a covariance-regularisation term :math:`\lambda`.
A greedy dynamic-programming search finds an approximate solution in linear time.

The maximum number of change points ``k_max`` is an upper bound; the algorithm
may return fewer if additional splits do not improve the regularised likelihood.

| **Type:** state detection
| **Supervision:** unsupervised or semi-supervised
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
   * - ``k_max``
     - int
     - ``10``
     - Maximum number of change points.
   * - ``lamb``
     - float
     - ``1.0``
     - Regularisation :math:`\lambda` (>= 0).  Controls covariance shrinkage.
   * - ``max_shuffles``
     - int
     - ``250``
     - Maximum number of shuffles during the greedy search.
   * - ``verbose``
     - bool
     - ``False``
     - Print progress information.
   * - ``random_state``
     - int / None
     - ``None``
     - Random seed.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import GreedyGaussianDetector

   detector = GreedyGaussianDetector(k_max=8, lamb=1.0)
   labels = detector.fit_predict(X)

**Implementation:** Adapted from aeon.  BSD 3-Clause.

**Reference:** Hallac, Nystrup & Boyd (2019), *Greedy Gaussian Segmentation of
Multivariate Time Series*, Advances in Data Analysis and Classification.

Submodules
----------

tsseg.algorithms.ggs.detector module
------------------------------------

.. automodule:: tsseg.algorithms.ggs.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.ggs
   :members:
   :show-inheritance:
   :undoc-members:
