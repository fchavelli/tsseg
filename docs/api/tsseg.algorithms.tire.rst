tsseg.algorithms.tire package
=============================

TIRE — Time-Invariant Representation for change point detection.

Description
-----------

TIRE learns a compact representation by training ensembles of parallel
autoencoders in both the time domain (TD) and frequency domain (FD).  The
*shared* latent dimensions across autoencoders capture the time-invariant signal
statistics while the *private* dimensions capture time-varying noise.  A
dissimilarity curve is computed from the shared representations and peaks in
this curve indicate change points.

By tuning the loss weights and latent dimensions for each domain you can
control whether the detector focuses on amplitude changes (TD) or spectral
changes (FD) or both.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
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
     - ``20``
     - Sliding window size (>= 4).
   * - ``stride``
     - int
     - ``1``
     - Stride between consecutive windows.
   * - ``domain``
     - str
     - ``"both"``
     - Domain(s) for representation learning (``"TD"``, ``"FD"``, ``"both"``).
   * - ``intermediate_dim_td``
     - int
     - ``0``
     - Hidden dim of TD autoencoder (0 = skip intermediate layer).
   * - ``latent_dim_td``
     - int
     - ``1``
     - Latent dim of TD autoencoder.
   * - ``nr_shared_td``
     - int
     - ``1``
     - Number of shared latent dims (TD).
   * - ``nr_ae_td``
     - int
     - ``3``
     - Number of parallel TD autoencoders.
   * - ``loss_weight_td``
     - float
     - ``1.0``
     - Loss weight for TD component.
   * - ``intermediate_dim_fd``
     - int
     - ``10``
     - Hidden dim of FD autoencoder.
   * - ``latent_dim_fd``
     - int
     - ``1``
     - Latent dim of FD autoencoder.
   * - ``nr_shared_fd``
     - int
     - ``1``
     - Number of shared latent dims (FD).
   * - ``nr_ae_fd``
     - int
     - ``3``
     - Number of parallel FD autoencoders.
   * - ``loss_weight_fd``
     - float
     - ``1.0``
     - Loss weight for FD component.
   * - ``nfft``
     - int
     - ``30``
     - FFT size for frequency-domain windows.
   * - ``norm_mode``
     - str
     - ``"timeseries"``
     - Normalisation scope (``"window"`` or ``"timeseries"``).
   * - ``peak_distance_fraction``
     - float
     - ``0.01``
     - Min fraction of series length between peaks.
   * - ``max_epochs``
     - int
     - ``20``
     - Maximum training epochs.
   * - ``patience``
     - int
     - ``5``
     - Early-stopping patience.
   * - ``learning_rate``
     - float
     - ``1e-3``
     - Optimiser learning rate.
   * - ``n_segments``
     - int / None
     - ``None``
     - Number of segments (overrides peak detection).
   * - ``axis``
     - int
     - ``0``
     - Time axis.
   * - ``random_state``
     - int / None
     - ``None``
     - Random seed.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import TireDetector

   detector = TireDetector(window_size=30, domain="both", n_segments=5)
   labels = detector.fit_predict(X)

**Implementation:** *Origin: new code.*

**Reference:** De Ryck, De Vos, Bertrand & Verhoest (2021), *Change Point
Detection in Time Series Data Using Autoencoders with a Time-Invariant
Representation*, IEEE TSIPN.

Submodules
----------

tsseg.algorithms.tire.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.tire.detector
   :members:
   :show-inheritance:
   :undoc-members:

tsseg.algorithms.tire.utils module
----------------------------------

.. automodule:: tsseg.algorithms.tire.utils
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.tire
   :members:
   :show-inheritance:
   :undoc-members:
