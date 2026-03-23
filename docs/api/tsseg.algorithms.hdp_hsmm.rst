tsseg.algorithms.hdp\_hsmm package
==================================

HDP-HSMM — Bayesian non-parametric state detection with Gibbs sampling.

Description
-----------

HDP-HSMM (Hierarchical Dirichlet Process Hidden Semi-Markov Model) is a
non-parametric Bayesian model for time series segmentation that *does not
require the number of states to be specified in advance*.  It extends the HMM
by modelling arbitrary (non-geometric) state durations through explicit duration
distributions.

The generative model:

- A **Hierarchical Dirichlet Process** draws an infinite discrete distribution
  over states (truncated at ``n_max_states``).  Concentration parameters
  ``alpha`` and ``gamma`` control state reuse.
- Each state has **Normal-Inverse-Wishart** emission parameters, allowing
  multivariate Gaussian observations with learned mean and covariance.
- Each state has a **Negative-Binomial** duration distribution (shape
  ``dur_alpha``, rate ``dur_beta``).
- Inference uses blocked Gibbs sampling over ``n_iter`` iterations.

| **Type:** state detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 58

   * - Name
     - Type
     - Default
     - Description
   * - ``alpha``
     - float
     - ``6.0``
     - Concentration for the DP prior on transitions.
   * - ``gamma``
     - float
     - ``6.0``
     - Concentration for the top-level DP.
   * - ``init_state_concentration``
     - float
     - ``6.0``
     - Concentration for the initial state distribution.
   * - ``n_iter``
     - int
     - ``200``
     - Number of Gibbs sampling iterations.
   * - ``n_max_states``
     - int
     - ``20``
     - Truncation level (max states).
   * - ``trunc``
     - int
     - ``100``
     - Truncation level for duration distributions.
   * - ``kappa0``
     - float
     - ``0.25``
     - Prior strength for NIW.
   * - ``nu0``
     - float / None
     - ``None``
     - Degrees of freedom for NIW (default: ``obs_dim + 2``).
   * - ``prior_mean``
     - float / array
     - ``0.0``
     - Prior mean for emissions.
   * - ``prior_scale``
     - float / array
     - ``1.0``
     - Scale matrix for NIW.
   * - ``dur_alpha``
     - float
     - ``2.0``
     - Shape for duration Gamma prior.
   * - ``dur_beta``
     - float
     - ``0.1``
     - Rate for duration Gamma prior.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import HdpHsmmDetector

   detector = HdpHsmmDetector(n_iter=200, n_max_states=10)
   states = detector.fit_predict(X)

**Implementation:** Pure NumPy/SciPy Gibbs sampler.  *Origin: new code.*  Replaces
the earlier ``pyhsmm``-backed implementation.

**Reference:** Johnson & Willsky (2013), *Bayesian Nonparametric Hidden
Semi-Markov Models*, JMLR; Nagano, Nakamura, Nagai, Mochihashi, Kobayashi &
Kaneko (2019), *Sequence Pattern Extraction by Segmenting Time Series Data
Using GP-HSMM with HDP*, IEEE RA-L.

Submodules
----------

tsseg.algorithms.hdp\_hsmm.detector module
------------------------------------------

.. automodule:: tsseg.algorithms.hdp_hsmm.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.hdp_hsmm
   :members:
   :show-inheritance:
   :undoc-members:
