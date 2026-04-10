tsseg.algorithms.vqtss package
==============================

VQ-TSS — Vector Quantized Time Series Segmentation.

Description
-----------

VQ-TSS is a predictive segmentation model that learns discrete state codes via
a VQ-VAE bottleneck:

1. **Encoder** — dilated residual convolutions map sliding windows to a
   continuous latent space.
2. **Vector Quantisation** — an EMA-updated codebook discretises the latent
   vectors into a finite set of state codes.
3. **Predictor** — an InfoNCE objective trains the predictor to match the
   next-step continuous latent, encouraging the codebook to capture semantically
   meaningful states.

A temporal-smoothness regulariser discourages rapid label switching.

| **Type:** state detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Requires:** PyTorch

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
     - Sliding-window length for training.
   * - ``stride``
     - int
     - ``1``
     - Stride for sliding-window extraction.
   * - ``hidden_dim``
     - int
     - ``64``
     - Latent / codebook dimension.
   * - ``num_embeddings``
     - int
     - ``64``
     - Number of VQ codebook entries (max discrete states).
   * - ``commitment_cost``
     - float
     - ``0.25``
     - VQ commitment-loss coefficient.
   * - ``decay``
     - float
     - ``0.99``
     - EMA decay for codebook updates.
   * - ``smoothness_weight``
     - float
     - ``0.1``
     - Temporal-smoothness regularisation weight.
   * - ``contrastive_temperature``
     - float
     - ``0.07``
     - Temperature for InfoNCE logits.
   * - ``neg_temporal_margin``
     - int
     - ``5``
     - Timesteps within +/- margin masked from negatives.
   * - ``learning_rate``
     - float
     - ``1e-3``
     - Adam learning rate.
   * - ``batch_size``
     - int
     - ``64``
     - Mini-batch size.
   * - ``epochs``
     - int
     - ``10``
     - Training epochs.
   * - ``max_grad_norm``
     - float
     - ``1.0``
     - Gradient clipping (0 = disable).
   * - ``device``
     - str / None
     - ``None``
     - PyTorch device (auto-detected if ``None``).
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

   from tsseg.algorithms import VQTSSDetector

   detector = VQTSSDetector(window_size=128, num_embeddings=32, epochs=20)
   states = detector.fit_predict(X)

**Implementation:** *Origin: new code.*

**Reference:** —

Submodules
----------

tsseg.algorithms.vqtss.detector module
--------------------------------------

.. automodule:: tsseg.algorithms.vqtss.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.vqtss
   :show-inheritance:
