E2USD
=====

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

Each window gets one state, so ``window_size`` sets the time resolution: a
window that straddles a change point can be embedded next to either of the two
segments, not always the one that covers more of it. Keep the window well
below the length of the shortest segment; the authors' experiments use windows
of 128 to 512 points. On a 1000-point series of three Gaussian segments of
300, 400 and 300 points, only 5 of the 15 windows of the default 256 points
lie inside a single segment, and the adjusted Rand index is 0.45 with
``n_states=3`` (0.28 to 0.52 over five seeds); it is 0.83 to 0.95 with
``window_size=32, step=32``, and 0.90 to 0.91 with the default window when the
segments are ten times longer.

| **Type:** state detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n\,\ell\log\ell)` per pass over the sliding
  windows, where :math:`\ell` is ``window_size``
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
     - Size of the sliding window; must be even.
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
     - Concentration parameter for DPGMM.
   * - ``batch_size``
     - int
     - ``1``
     - No effect: the training set is one sample, the whole series.
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
     - No effect: the encoder network does not use it.
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
     - Kernel size of the two convolutions; at most ``min(window_size, 64)``.
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

   from tsseg.algorithms import E2USDDetector

   detector = E2USDDetector(window_size=256, n_states=10, nb_steps=20)
   states = detector.fit_predict(X)

**Implementation:** Adapted from `AI4CTS/E2Usd <https://github.com/AI4CTS/E2Usd>`_.
No licence found in original repository. Two parts come from projects with a
published licence: the ``Dataset`` class from the code of Franceschi et al.
(2019), `White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
<https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries>`_,
Apache License 2.0, and the ``moving_avg`` and ``series_decomp`` blocks from
`thuml/Autoformer <https://github.com/thuml/Autoformer>`_, MIT licence. The
``LICENSE`` file of ``tsseg/algorithms/e2usd`` holds these two licences and
covers these parts only.

**Reference:** Lai, Li, Zhang, Zhao, Qian & Jensen (2024), *E2Usd:
Efficient-yet-effective Unsupervised State Detection for Multivariate Time
Series*, Proceedings of the ACM Web Conference 2024 (WWW), pp. 3010–3021,
doi:10.1145/3589334.3645593.

API reference
-------------

.. automodule:: tsseg.algorithms.e2usd.detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.e2usd.e2usd
   :members:
   :show-inheritance:
   :undoc-members:
