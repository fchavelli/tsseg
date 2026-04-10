tsseg.algorithms.tscp2 package
==============================

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
segment.  A similarity curve is built from all adjacent-window pairs and
thresholded (or top-*K* peaks selected) to produce change points.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
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
     - Number of change points (``None`` = auto-threshold).
   * - ``similarity_threshold``
     - float / None
     - ``None``
     - Custom similarity threshold for detecting CPs.
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
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import TSCP2Detector

   detector = TSCP2Detector(window_size=128, n_cps=5, epochs=50)
   labels = detector.fit_predict(X)

**Implementation:** TensorFlow reimplementation adapted from original code by
Deldari et al.

**Reference:** Deldari, Smith, Xue & Salim (2021), *Time Series Change Point
Detection with Self-Supervised Contrastive Predictive Coding*, WWW.

Submodules
----------

tsseg.algorithms.tscp2.detector module
--------------------------------------

.. automodule:: tsseg.algorithms.tscp2.detector
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.tscp2.losses module
------------------------------------

.. automodule:: tsseg.algorithms.tscp2.losses
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.tscp2.network module
-------------------------------------

.. automodule:: tsseg.algorithms.tscp2.network
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.tscp2
   :show-inheritance:
