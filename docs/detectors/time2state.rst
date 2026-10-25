Time2State
==========

Time2State — unsupervised latent state inference.

Description
-----------

Time2State infers latent states in time series data using a Causal CNN-based
encoder and a novel unsupervised loss function (LSE-Loss).  A sliding window
extracts subsequences; the encoder maps them to a low-dimensional latent space
where a Dirichlet Process Gaussian Mixture Model (DPGMM) clusters the
embeddings into states without requiring the number of states *a priori*.

The framework drastically reduces computational cost compared to operating on
raw time series by compressing the representation before clustering.

| **Type:** state detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n\,\ell)` per pass over the sliding windows,
  where :math:`\ell` is ``window_size``
| **Requires:** PyTorch

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
     - ``256``
     - Sliding window size; must be even.
   * - ``step``
     - int
     - ``50``
     - Step size of the sliding window.
   * - ``n_states``
     - int
     - ``20``
     - Upper bound on the number of states (DPGMM components).
   * - ``alpha``
     - float
     - ``1e3``
     - DPGMM concentration parameter.
   * - ``batch_size``
     - int
     - ``1``
     - No effect: the training set is one sample, the whole series.
   * - ``nb_steps``
     - int
     - ``20``
     - Training optimisation steps.
   * - ``lr``
     - float
     - ``0.003``
     - Learning rate.
   * - ``depth``
     - int
     - ``10``
     - Depth of the Causal CNN.
   * - ``out_channels``
     - int
     - ``4``
     - Encoder output channels.
   * - ``reduced_size``
     - int
     - ``80``
     - CNN output dimension before the linear layer.
   * - ``kernel_size``
     - int
     - ``3``
     - Convolution kernel size.
   * - ``use_gpu``
     - bool / None
     - ``None``
     - Run on CUDA device 0 (``None``: when available).
   * - ``random_state``
     - int / None
     - ``None``
     - Random seed.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import Time2StateDetector

   detector = Time2StateDetector(window_size=128, n_states=10)
   states = detector.fit_predict(X)

**Implementation:** Adapted from `Lab-ANT/Time2State
<https://github.com/Lab-ANT/Time2State>`_, MIT licence. Its causal CNN encoder
and ``Dataset`` class come from the code of Franceschi et al. (2019),
`White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
<https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries>`_,
Apache License 2.0, and partly from `locuslab/TCN
<https://github.com/locuslab/TCN>`_, MIT licence. The ``LICENSE`` file of
``tsseg/algorithms/time2state`` holds the three licences.

**Reference:** Wang, Wu, Zhou & Cai (2023), *Time2State: An Unsupervised
Framework for Inferring the Latent States in Time Series Data*, SIGMOD.

API reference
-------------

.. automodule:: tsseg.algorithms.time2state.detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.time2state.time2state
   :members:
   :show-inheritance:
   :undoc-members:
