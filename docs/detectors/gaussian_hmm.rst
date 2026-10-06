Gaussian HMM
============

Hidden Markov model with Gaussian emissions, fitted by expectation-maximisation.

Description
-----------

Each hidden state :math:`k` emits a Gaussian :math:`\mathcal{N}(\mu_k,
\operatorname{diag}(\sigma_k^2))`, and the states follow a first-order Markov
chain with initial probabilities :math:`\pi` and transition matrix :math:`A`.
All parameters are estimated by maximum likelihood with the Baum-Welch
algorithm, an expectation-maximisation procedure whose E-step is the
forward-backward algorithm (Rabiner, 1989). Each time step then receives the
state of the most probable path (Viterbi).

- **Initialisation**: k-means on the (z-normalised) observations gives the means
  and variances; :math:`\pi` and :math:`A` start uniform. ``n_init``
  initialisations with different seeds are run and the one with the highest
  log-likelihood is kept.
- **Regularisation**: ``reg_covar`` (positive) is added to every variance at
  each M-step, which limits the collapse of a state onto a few points. On integer
  or coarsely quantised data a state can still collapse onto a single value;
  raise ``reg_covar`` then.
- **Number of states**: given by ``n_states``, or, with ``n_states=None``,
  chosen by the Bayesian information criterion among ``1, ..., max_states``.
  The fitted model has :math:`(K-1) + K(K-1) + 2Kd` free parameters.

The emissions are independent given the state, so the model has no notion of
duration or of temporal shape within a state. It suits regimes that differ in
level or spread, not regimes that differ only in frequency or waveform: on
oscillating signals the states follow the oscillation and the labels change
within a regime. For the same reason BIC often selects ``max_states`` on long
autocorrelated series (a warning says so), and the number of states is then
better given.

| **Type:** state detection
| **Supervision:** unsupervised (BIC) or semi-supervised (``n_states``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(nK^2 + nKd)` per EM iteration; the BIC search
  multiplies it by the number of candidates
| **Requires:** nothing beyond the core dependencies; numba
  (``tsseg[accelerators]``) speeds up the recursions considerably

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 14 12 52

   * - Name
     - Type
     - Default
     - Description
   * - ``n_states``
     - int / None
     - ``None``
     - Number of hidden states; ``None`` selects it by BIC.
   * - ``max_states``
     - int
     - ``8``
     - Largest number of states tried by BIC.
   * - ``n_iter``
     - int
     - ``100``
     - Maximum number of EM iterations per initialisation.
   * - ``tol``
     - float
     - ``1e-4``
     - EM stops when the log-likelihood per time step improves by less.
   * - ``n_init``
     - int
     - ``3``
     - Number of k-means initialisations.
   * - ``reg_covar``
     - float
     - ``1e-3``
     - Added to the variances at each M-step (positive; units of the normalised
       data).
   * - ``normalize``
     - bool
     - ``True``
     - Z-normalise each channel before fitting.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed of the k-means initialisations.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import GaussianHMMDetector

   # number of states chosen by BIC
   labels = GaussianHMMDetector(random_state=0).fit_predict(X)

   # known number of states
   labels = GaussianHMMDetector(n_states=4, random_state=0).fit_predict(X)

**Implementation:** written for tsseg in NumPy, with numba for the
forward-backward and Viterbi recursions. With the same starting point, the
parameters and log-likelihood after each EM iteration match those of
`hmmlearn <https://github.com/hmmlearn/hmmlearn>`_ (``GaussianHMM`` with
``covariance_type="diag"`` and ``covars_prior=0``) to :math:`10^{-10}` or
better.
It replaces :doc:`hmm`, which only decoded with user-supplied parameters.

**Reference:** Rabiner (1989), *A tutorial on hidden Markov models and selected
applications in speech recognition*, Proceedings of the IEEE 77(2); Schwarz
(1978), *Estimating the dimension of a model*, The Annals of Statistics 6(2).

API reference
-------------

.. automodule:: tsseg.algorithms.gaussian_hmm.detector
   :members:
   :show-inheritance:
   :undoc-members:
