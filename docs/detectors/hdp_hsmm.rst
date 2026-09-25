HDP-HSMM
========

HDP-HSMM — Bayesian non-parametric state detection with Gibbs sampling.

Description
-----------

HDP-HSMM (Hierarchical Dirichlet Process Hidden Semi-Markov Model) is a
non-parametric Bayesian model for time series segmentation that *does not
require the number of states to be specified in advance*.  It extends the HMM
by modelling arbitrary (non-geometric) state durations through explicit duration
distributions.

The generative model (Johnson & Willsky, 2013):

- A **Hierarchical Dirichlet Process** (weak limit, ``n_max_states`` states) draws the
  global state weights and the transition distributions; self-transitions are removed,
  durations being explicit.  ``alpha`` and ``gamma`` control state reuse.
- Each state has **Normal-Inverse-Wishart** emission parameters, allowing
  multivariate Gaussian observations with learned mean and covariance.
- Each state has a shifted **Poisson** duration distribution with a Gamma prior on its
  rate (negative binomial optional).
- Inference is blocked Gibbs sampling over ``n_iter`` sweeps, as in the authors'
  ``pyhsmm``; the segmentation returned is the last sample.

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
     - Concentration of the transition distributions around the global weights.
   * - ``gamma``
     - float
     - ``6.0``
     - Concentration of the global state weights.
   * - ``init_state_concentration``
     - float
     - ``6.0``
     - Concentration of the initial state distribution.
   * - ``n_iter``
     - int
     - ``20``
     - Number of Gibbs sweeps (quality is flat from the second sweep).
   * - ``n_max_states``
     - int
     - ``20``
     - Weak-limit truncation of the number of states.
   * - ``trunc``
     - int / None
     - ``None``
     - Maximum segment duration in samples; ``None``: no truncation.
   * - ``kappa0``
     - float
     - ``0.25``
     - NIW pseudo-count on the mean.
   * - ``nu0``
     - float / None
     - ``None``
     - NIW degrees of freedom (default: ``d + 1 + emission_strength``).
   * - ``prior_mean``
     - float / array
     - ``0.0``
     - NIW prior mean.
   * - ``prior_scale``
     - float / array
     - ``1.0``
     - Prior expected covariance when ``nu0`` is None, else the inverse-Wishart scale.
   * - ``emission_strength``
     - float
     - ``100.0``
     - Pseudo-observations behind the prior covariance.
   * - ``dur_family``
     - str
     - ``"poisson"``
     - Duration distribution, ``"poisson"`` or ``"negbin"``.
   * - ``dur_alpha``
     - float
     - ``5.0``
     - Strength of the duration prior (Gamma shape for the Poisson rate).
   * - ``dur_beta``
     - float / None
     - ``None``
     - Gamma rate of the Poisson prior; ``None``: prior mean duration of T/3 (see below).
   * - ``dur_r``
     - float
     - ``1.0``
     - Negative binomial: fixed number of failures.
   * - ``max_len``
     - int / None
     - ``2000``
     - Longer series are reduced by block means before the fit.
   * - ``block_features``
     - str
     - ``"mean"``
     - Block features when reducing: ``"mean"`` or ``"meanstd"``.
   * - ``normalize``
     - bool
     - ``True``
     - z-normalise each channel before the fit.
   * - ``init``
     - str
     - ``"states"``
     - Chain start: ``"states"`` (drawn from the prior HSMM, as pyhsmm) or ``"params"``.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import HdpHsmmDetector

   detector = HdpHsmmDetector()
   states = detector.fit_predict(X)

**Implementation:** NumPy/SciPy Gibbs sampler, forward messages compiled with numba
when it is installed (``tsseg[accelerators]``; otherwise interpreted, about 100 times
slower), checked against ``pyhsmm`` and against exact enumeration.  *Origin: new code.*
Replaces the earlier ``pyhsmm``-backed implementation and the
first NumPy implementation (``HdpHsmmDetectorV1``, deprecated).

**References:** Johnson & Willsky (2013), *Bayesian Nonparametric Hidden Semi-Markov
Models*, JMLR 14, for the model and the sampler implemented here; Johnson & Willsky
(2010), *The Hierarchical Dirichlet Process Hidden Semi-Markov Model*, UAI, the
conference paper that introduced the model.

API reference
-------------

.. automodule:: tsseg.algorithms.hdp_hsmm.detector
   :members:
   :show-inheritance:
   :undoc-members:
