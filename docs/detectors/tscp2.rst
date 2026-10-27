TS-CP2
======

TS-CP² — change point detection with contrastive predictive coding.

Description
-----------

TS-CP² learns temporal representations using a Temporal Convolutional Network
(TCN) trained with a contrastive predictive coding (CPC) objective.  Pairs of
consecutive windows are encoded; if their embeddings are similar the boundary
between them is unlikely to be a change point, while dissimilar embeddings
signal a regime change.

The contrastive loss (configurable: NCE, debiased CL, focal CL, hard-negative
CL) encourages the encoder to capture the distributional structure of each
segment.  A similarity curve is built from all adjacent-window pairs. A
change point lowers the similarity of every pair centred less than one window
from it, so one change point is kept per dip: the pair centres are visited
from the least similar, and a centre less than one window from a change point
already kept is skipped. The candidates are all the centres when the number
of change points is given (the first ``n_cps`` kept), and the centres below
the similarity threshold otherwise.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n\,\ell\,F\,(d + F + \ell)/s)` per training epoch,
  where :math:`\ell` is ``window_size``, *s* ``stride`` and *F* ``nb_filters``
  (kernel size and depth of the TCN fixed): the encoder runs on both windows
  of the :math:`n/s` window pairs, and its first dense layer maps the
  :math:`\ell\,F` outputs of the TCN to about :math:`\ell` units (default
  ``dense_units``). Prediction costs one such pass. On a CPU, the time hardly
  grows with :math:`\ell` up to 128 or with *d* up to 64, and is about
  proportional to the number of batches: 5 to 8 s per epoch for 4,000 points
  (14 cores)
| **Requires:** TensorFlow and ``tcn``

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 55

   * - Name
     - Type
     - Default
     - Description
   * - ``window_size``
     - int
     - ``128``
     - Sliding window size for building window pairs.
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points, at least one window apart when the series
       leaves room for them (``None`` = use the similarity threshold).
   * - ``similarity_threshold``
     - float / None
     - ``None``
     - Similarity below which a pair centre is a candidate change point
       (``None`` = set by ``fit``).
   * - ``stride``
     - int
     - ``5``
     - Step between successive window pairs.
   * - ``code_size``
     - int
     - ``32``
     - Dimension of the learned representation.
   * - ``nb_filters``
     - int
     - ``64``
     - Number of TCN filters.
   * - ``kernel_size``
     - int
     - ``4``
     - TCN kernel size.
   * - ``dilations``
     - tuple
     - ``(1,2,4,8)``
     - Dilation factors for TCN layers.
   * - ``nb_stacks``
     - int
     - ``2``
     - Number of TCN stacks.
   * - ``dropout_rate``
     - float
     - ``0.0``
     - Dropout rate.
   * - ``batch_size``
     - int
     - ``64``
     - Training batch size.
   * - ``epochs``
     - int
     - ``100``
     - Training epochs.
   * - ``learning_rate``
     - float
     - ``1e-3``
     - Optimiser learning rate.
   * - ``loss``
     - str
     - ``"nce"``
     - Contrastive loss (``"nce"``, ``"dcl"``, ``"fc"``, ``"harddcl"``).
   * - ``temperature``
     - float
     - ``0.1``
     - Temperature for contrastive logits.
   * - ``similarity``
     - str
     - ``"cosine"``
     - Similarity function.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed of the weight initialisation, the shuffling of the training
       windows and the dropout masks (``None`` = a different model at each
       fit). Reseeds the global Python, NumPy and TensorFlow generators.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import TSCP2Detector

   detector = TSCP2Detector(window_size=128, n_cps=5, epochs=50)
   change_points = detector.fit_predict(X)

**Implementation:** TensorFlow reimplementation adapted from original code by
Deldari et al.

**Reference:** Deldari, Smith, Xue & Salim (2021), *Time Series Change Point
Detection with Self-Supervised Contrastive Predictive Coding*, WWW.

API reference
-------------

.. automodule:: tsseg.algorithms.tscp2.detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.tscp2.losses
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.tscp2.network
   :members:
   :show-inheritance:
   :undoc-members:
