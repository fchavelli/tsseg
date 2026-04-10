tsseg.algorithms.e2usd package
==============================

E2USD — Efficient-yet-Effective Unsupervised State Detection.

Description
-----------

E2USD combines two key ideas for scalable, accurate unsupervised state detection
in multivariate time series:

1. **DDEM** (Decomposed Dual-view Embedding Module) — a lightweight encoder
   that compresses sliding windows into low-dimensional representations using
   FFT-based compression and a decomposed contrastive learning objective.
2. **DPGMM clustering** — a Dirichlet Process Gaussian Mixture Model that
   automatically determines the number of states from the learned embeddings.

A False Negative Cancellation Contrastive Learning method (FNCCLearning) is
used to counteract false negatives and produce cluster-friendly embedding spaces.

| **Type:** state detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** multivariate
| **Requires:** PyTorch

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``window_size``
     - int
     - ``256``
     - Size of the sliding window.
   * - ``step``
     - int
     - ``50``
     - Step size of the sliding window.
   * - ``n_states``
     - int
     - ``20``
     - Maximum number of states for DPGMM clustering.
   * - ``alpha``
     - float
     - ``1e3``
     - Concentration parameter for DPGMM.
   * - ``batch_size``
     - int
     - ``1``
     - Training batch size.
   * - ``nb_steps``
     - int
     - ``20``
     - Number of optimisation steps.
   * - ``lr``
     - float
     - ``0.003``
     - Learning rate.
   * - ``depth``
     - int
     - ``1``
     - Depth of the DDEM encoder network.
   * - ``out_channels``
     - int
     - ``4``
     - Number of output channels of the encoder.
   * - ``reduced_size``
     - int
     - ``80``
     - Dimension of the CNN output before the final linear layer.
   * - ``kernel_size``
     - int
     - ``3``
     - Kernel size for CNN convolutions.
   * - ``use_gpu``
     - bool / None
     - ``None``
     - Force GPU usage (``None`` = auto-detect).
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

   from tsseg.algorithms import E2USDDetector

   detector = E2USDDetector(window_size=256, n_states=10, nb_steps=20)
   states = detector.fit_predict(X)

**Implementation:** Adapted from `AI4CTS/E2Usd <https://github.com/AI4CTS/E2Usd>`_.
No licence found in original repository.

**Reference:** Lai, Zhao, Li, Qian, Zhang & Jensen (2024), *E2Usd:
Efficient-yet-effective Unsupervised State Detection for Multivariate Time
Series*, WWW.

Submodules
----------

tsseg.algorithms.e2usd.detector module
--------------------------------------

.. automodule:: tsseg.algorithms.e2usd.detector
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.e2usd.e2usd module
-----------------------------------

.. automodule:: tsseg.algorithms.e2usd.e2usd
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.e2usd
   :show-inheritance:
