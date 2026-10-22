ESPRESSO
========

ESPRESSO — Entropy and Shape-aware Time Series Segmentation.

Description
-----------

ESPRESSO combines the temporal shape of each channel, read from its matrix
profile, with an entropy criterion computed over all channels to segment
multi-dimensional time series:

1. For each channel, compute the matrix profile of the subsequences of length
   ``window_size`` (z-normalised Euclidean distance, FFT-based self-join written
   in NumPy; ``stumpy`` is not used). It links each subsequence to its nearest
   neighbour by an arc.
2. Extend these arcs along chains of nearest neighbours for at most
   ``chain_len`` rounds, and add to every position an arc spans a weight that
   decreases with the arc length. The result is the channel's semantic density
   curve: few arcs cross a segment boundary.
3. Smooth each density curve and take its local minima, at least
   ``peak_distance_fraction`` times the series length apart and ranked by
   prominence, as candidate change points.
4. Among the candidates of each channel, greedily add the one that
   maximises the information gain (entropy reduction, as in :doc:`IGTS <igts>`) of the
   segmentation of all channels. The change points are the first
   ``n_segments - 1`` candidates picked, from the channel whose picks reach the
   highest gain; there are fewer when that channel has fewer candidates.

| **Type:** change point detection
| **Supervision:** semi-supervised (``n_segments`` required)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n^{2}\log n)`: one matrix profile per channel,
  with an FFT for each subsequence

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
     - ``64``
     - Subsequence length for the matrix profile (>= 4, at most half the
       series length).
   * - ``chain_len``
     - int
     - ``3``
     - Iterations for expanding arc sets.
   * - ``n_segments``
     - int / None
     - ``None``
     - Target number of segments (``1`` = no change point); required.
   * - ``peak_distance_fraction``
     - float
     - ``0.01``
     - Minimum spacing between peaks as a fraction of the series length.
   * - ``axis``
     - int
     - ``0``
     - Time axis.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed of the order in which subsequences are visited for the matrix
       profile; it changes the result only when distances tie (exactly
       repeated patterns). ``None`` draws a fresh seed at each ``predict``.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import EspressoDetector

   detector = EspressoDetector(window_size=64, chain_len=5, n_segments=4)
   change_points = detector.fit_predict(X)

**Implementation:** Python translation of the authors' MATLAB code
(`cruiseresearchgroup/ESPRESSO <https://github.com/cruiseresearchgroup/ESPRESSO>`_),
which is not shipped with tsseg. No licence found in the original repository.
*Origin: translated from the original MATLAB code.*

**Reference:** Deldari, Smith, Sadri & Salim (2020), *ESPRESSO: Entropy and
ShaPe awaRe timE-Series SegmentatiOn for Processing Heterogeneous Sensor Data*,
Proc. ACM IMWUT 4(3) (UbiComp 2020).

API reference
-------------

.. automodule:: tsseg.algorithms.espresso.detector
   :members:
   :show-inheritance:
   :undoc-members:
