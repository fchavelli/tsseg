TIRE
====

TIRE – Time-Invariant Representation for change point detection.

Description
-----------

TIRE learns a compact representation by training ensembles of parallel
autoencoders in both the time domain (TD) and frequency domain (FD).  The
*shared* latent dimensions across autoencoders capture the time-invariant signal
statistics while the *private* dimensions capture time-varying noise.  A
dissimilarity curve is computed from the shared representations and peaks in
this curve indicate change points: the ``n_segments - 1`` most prominent ones
when ``n_segments`` is given, otherwise every peak whose prominence, relative
to the highest one, reaches ``prominence_threshold``.

Each channel is rescaled to :math:`[-1, 1]` on the whole series before the
windows are cut (a constant channel is set to 0; a constant series has no
change point). The frequency domain input of a window is the modulus of the
discrete Fourier transform of each channel, rescaled channel by channel and
concatenated. The shared features are smoothed by a triangular moving average
in every domain before the dissimilarity is computed.

By tuning the loss weights and latent dimensions for each domain you can
control whether the detector focuses on amplitude changes (TD) or spectral
changes (FD) or both.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n\,\ell)` per training epoch, where
  :math:`\ell` is ``window_size``
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
     - DFT size of the window of each channel (longer windows are cut).
   * - ``norm_mode``
     - str
     - ``"timeseries"``
     - Mean removed before the DFT, per channel: of each window (``"window"``)
       or of the series (``"timeseries"``).
   * - ``peak_distance_fraction``
     - float
     - ``0.0``
     - Minimum distance between two change points, as a fraction of the
       series length (0: no constraint).
   * - ``prominence_threshold``
     - float
     - ``0.5``
     - Minimum prominence of a peak, relative to the highest one, when
       ``n_segments`` is ``None``.
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
   change_points = detector.fit_predict(X)

**Implementation:** PyTorch reimplementation of the authors' TensorFlow/Keras
code (`deryckt/TIRE <https://github.com/deryckt/TIRE>`_): the parallel
autoencoders and their training follow ``TIRE.py``, and the helpers of
``tsseg.algorithms.tire.utils`` are copied from its ``utils.py``.  No licence
found in the original repository.  The original code returns a change score
per time step; picking change points from it (``n_segments``,
``prominence_threshold``) is specific to tsseg.  On multivariate series tsseg
follows the paper where the original code does not: it transforms the window
of each channel (eq. 3) instead of the concatenated windows cut to ``nfft``
points, and it rescales each channel (§III-B) instead of leaving this to the
caller.  *Origin: reimplemented from
the original code.*

**Reference:** De Ryck, De Vos & Bertrand (2021), *Change Point Detection in
Time Series Data Using Autoencoders with a Time-Invariant Representation*,
IEEE Transactions on Signal Processing, doi:10.1109/TSP.2021.3087031
(`arXiv:2008.09524 <https://arxiv.org/abs/2008.09524>`_).

API reference
-------------

.. automodule:: tsseg.algorithms.tire.detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.tire.utils
   :members:
   :show-inheritance:
   :undoc-members:
