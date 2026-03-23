tsseg.algorithms.hmm package
============================

HMM — Hidden Markov Model state annotation via Viterbi decoding.

Description
-----------

Annotates a univariate time series with hidden-state labels using the Viterbi
algorithm.  The emission distributions, transition matrix and initial
probabilities must be provided by the user (no EM learning).  This makes the
detector suitable as a baseline or when prior distributions are known.

| **Type:** state detection
| **Supervision:** requires known distributions (no learning)
| **Scope:** univariate only

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 54

   * - Name
     - Type
     - Default
     - Description
   * - ``emission_funcs``
     - list / None
     - ``None``
     - List of callables (PDFs) for each hidden state.  Default: two-state
       Gaussian ``N(0,1)`` / ``N(1,1)``.
   * - ``transition_prob_mat``
     - ndarray / None
     - ``None``
     - Row-stochastic transition matrix.  Default: ``[[0.9,0.1],[0.1,0.9]]``.
   * - ``initial_probs``
     - ndarray / None
     - ``None``
     - Initial state probabilities.  Default: uniform.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import HMMDetector

   detector = HMMDetector()   # default two-state Gaussian
   labels = detector.fit_predict(X)

**Implementation:** Adapted from aeon.  BSD 3-Clause.

**Reference:** Rabiner (1989), *A tutorial on hidden Markov models and selected
applications in speech recognition*, Proceedings of the IEEE.

Submodules
----------

tsseg.algorithms.hmm.detector module
------------------------------------

.. automodule:: tsseg.algorithms.hmm.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.hmm
   :members:
   :show-inheritance:
   :undoc-members:
