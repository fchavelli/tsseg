tsseg.algorithms.hidalgo package
================================

Hidalgo — Heterogeneous Intrinsic Dimensionality Algorithm.

Description
-----------

Hidalgo performs Bayesian clustering by estimating the local intrinsic
dimensionality of data manifolds.  It assigns each observation to one of
``K_states`` manifolds using Gibbs sampling, a Potts-model spatial prior and
nearest-neighbour distance statistics.

The algorithm is designed for high-dimensional data and is particularly suited
when different states occupy manifolds of different dimensionality.

| **Type:** state detection
| **Supervision:** semi-supervised (``K_states`` required)
| **Scope:** multivariate (uses nearest-neighbour distances)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 58

   * - Name
     - Type
     - Default
     - Description
   * - ``metric``
     - str / callable
     - ``"euclidean"``
     - Distance metric for sklearn ``NearestNeighbors``.
   * - ``K_states``
     - int
     - ``1``
     - Number of manifolds / states.
   * - ``zeta``
     - float
     - ``0.8``
     - Local homogeneity level, in :math:`(0, 1)`.
   * - ``q``
     - int
     - ``3``
     - Number of neighbours for local Z interaction.
   * - ``n_iter``
     - int
     - ``1000``
     - Number of Gibbs sampling iterations.
   * - ``n_replicas``
     - int
     - ``1``
     - Number of random restarts.
   * - ``burn_in``
     - float
     - ``0.9``
     - Fraction of iterations discarded as burn-in.
   * - ``fixed_Z``
     - bool
     - ``False``
     - Estimate parameters with fixed allocation Z.
   * - ``use_Potts``
     - bool
     - ``True``
     - Enable local Potts interaction between assignments.
   * - ``estimate_zeta``
     - bool
     - ``False``
     - Update zeta during sampling.
   * - ``sampling_rate``
     - int
     - ``10``
     - Save samples every *k* iterations.
   * - ``seed``
     - int
     - ``0``
     - Random seed.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import HidalgoDetector

   detector = HidalgoDetector(K_states=3, n_iter=500)
   states = detector.fit_predict(X)

**Implementation:** Adapted from aeon with numerical stability fix (log-domain
``sample_p``).  BSD 3-Clause.

**Reference:** Allegra, Facco, Denti, Laio & Mira (2020), *Data segmentation
based on the local intrinsic dimension*, Scientific Reports.

Submodules
----------

tsseg.algorithms.hidalgo.detector module
----------------------------------------

.. automodule:: tsseg.algorithms.hidalgo.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.hidalgo
   :show-inheritance:
