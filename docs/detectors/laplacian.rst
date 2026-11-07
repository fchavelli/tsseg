Laplacian segmentation
======================

State detection by spectral clustering of a recurrence graph of windows.

Description
-----------

McFee & Ellis (2014) find the repeated sections of a song (verse, chorus) with
a graph whose nodes are short frames of the recording. ``LaplacianDetector``
applies their structural decomposition to windows of a time series:

1. The series, standardised per channel, is cut into frames of
   ``window_size`` points every ``step`` points. Each frame is described by
   the log-magnitude spectrum of each channel (``feature="spec"``,
   :math:`\log(1 + |\mathrm{FFT}|)` with the signed log of the window sum,
   the DC term),
   which does not depend on the phase of a periodic pattern, or by the window
   itself (``feature="raw"``).
2. **Recurrence**: frames :math:`i` and :math:`j` are linked when each is
   among the :math:`k = 2\lceil\sqrt{N - 2e + 1}\rceil` nearest neighbours
   of the other (cosine distance; :math:`N` frames; links between frames
   closer than :math:`e = \lceil w / (2 s) \rceil` frames in time are
   ignored), with weight :math:`\exp(-d_{ij} / \tilde d)`, where
   :math:`\tilde d` is the median distance of a frame to its farthest link.
   A median filter of ``rec_smooth`` frames along the diagonals favours
   sequences of frames that repeat in the same order.
3. **Path**: consecutive frames are linked with weight
   :math:`\exp(-\|f_i - f_{i+1}\|^2 / \tilde\delta)`, where
   :math:`\tilde\delta` is the median of these squared distances.
4. The graph :math:`A = \mu R + (1 - \mu) P` balances the two kinds of edges:
   :math:`\mu` is the least-squares fit, frame by frame, of the degrees of
   :math:`\mu R` and :math:`(1 - \mu) P` (McFee & Ellis, Eqs. 6-7).
5. The smallest eigenvectors of the normalised Laplacian of :math:`A`, median
   filtered over ``evec_smooth`` frames and divided row-wise by their
   cumulative norm, are clustered by k-means into :math:`k` states. Each
   point takes the state of the frame whose centre is nearest.

A state is thus a set of frames that look alike wherever they occur in the
series (recurrence edges) and that stay together in time (path edges).

**Number of states.** Guided, :math:`k` is ``n_states``. Unsupervised, the
paper keeps, among :math:`m = 2, \dots, 10`, the partition of largest label
entropy whose mean segment lasts at least 10 s (Sec. 3.3), a threshold in
seconds without a counterpart for an arbitrary series; msaf and the librosa
example fix :math:`k`. tsseg uses
:math:`k = \mathrm{round}(\sqrt{k_L k_S})`, the geometric mean of two
estimates that err in opposite directions: :math:`k_L`, the number of Louvain
communities (modularity, resolution 1) of the recurrence graph with
:math:`2\lceil\sqrt N\rceil` neighbours, without the diagonal filter and
without path edges, which tends to over-segment, and :math:`k_S`, the
:math:`k \le` ``max_states`` whose k-means labels have the best cosine
silhouette in the space of the frame features, which tends to under-segment.
Both are kept in ``k_louvain_`` and ``k_silhouette_``.

**Symmetric recurrence.** The diagonal median filter of librosa does not keep
the recurrence matrix symmetric (its zero padding enters near the ends of the
diagonals). The librosa example and msaf decompose the resulting matrix with
``scipy.linalg.eigh``, which reads a single triangle, so the matrix they
decompose is not a graph Laplacian: its smallest eigenvalue can be negative,
and the solver can break down when a frame is almost isolated. tsseg
symmetrises the filtered matrix, :math:`(R_f + R_f^\top) / 2`, and checks that
the spectrum lies in :math:`[0, 2]`.

**Departures from the paper.** tsseg follows the librosa example and msaf
where they differ from the paper: cosine distance and
:math:`\exp(-d / \tilde d)` instead of a Gaussian kernel on the Euclidean
distance, :math:`2\lceil\sqrt{N - 2e + 1}\rceil` neighbours instead of
:math:`1 + \lceil 2 \log_2 N \rceil`, a median filter of the weighted
affinity instead of a majority vote over 17 frames on the binary graph, path
weights on the same features as the recurrence, and median-filtered
eigenvectors. The cosine ignores scale: with ``feature="raw"``, states that
differ only by their amplitude look alike, and with ``"spec"`` nearly
constant frames are dominated by their DC term, so that two levels of the
same sign look alike. Two frames at distance exactly 0 are never linked, and
when more than half of the consecutive frames are identical all path weights
are 1.

**Defaults.** ``feature="spec"``, ``window_size=128``, ``step=16`` were
declared before an evaluation on the 592 series of TSB-SEG. Guided, the mean
state matching score (SMS) is 0.717 per series, against 0.692 for CLaP, the
best state detector of TSB-SEG. It is above CLaP on five of the six
multivariate datasets and below it on the two univariate ones (TSSB: 0.697
against 0.898). Unsupervised, it is close to
CLaP on the mean over datasets and below it per series. The step is fixed in
points, so that the median filters span the same duration on every series;
there is no cap on the number of frames.

| **Type:** state detection
| **Supervision:** unsupervised or guided (``n_states``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(N^2 d w + N^3)` for :math:`N \approx n / s` frames
  of :math:`w` points and :math:`d` channels, :math:`O(N^2)` memory (dense
  graph); about 16 s and 0.9 GB for 56,000 points (3,500 frames,
  unsupervised, two threads)
| **Requires:** nothing beyond the core dependencies

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 14 18 46

   * - Name
     - Type
     - Default
     - Description
   * - ``n_states``
     - int / None
     - ``None``
     - Number of states; ``None`` estimates it (see above).
   * - ``window_size``
     - int
     - ``128``
     - Frame length :math:`w` in points; on a series shorter than
       :math:`4w`, frames of :math:`\max(8, \lfloor n / 4 \rfloor)` points.
   * - ``step``
     - int
     - ``16``
     - Points :math:`s` between consecutive frames. Cost grows as the cube
       of :math:`n / s`.
   * - ``feature``
     - str
     - ``"spec"``
     - ``"spec"`` (log-magnitude spectrum per channel) or ``"raw"`` (the
       window).
   * - ``rec_smooth``
     - int
     - ``9``
     - Median filter along the diagonals of the recurrence matrix, in frames
       (msaf default; 1: none).
   * - ``evec_smooth``
     - int
     - ``9``
     - Median filter of the eigenvectors along time, in frames (msaf default;
       1: none).
   * - ``max_states``
     - int
     - ``20``
     - Largest number of states of the unsupervised rule.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed of k-means, of the Louvain node order and of the frames sampled
       for the silhouette (at most 2,000).
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg import load_mocap
   from tsseg.algorithms import LaplacianDetector

   X, y = load_mocap(trial=0)
   labels = LaplacianDetector(n_states=4, random_state=0).fit_predict(X)

   detector = LaplacianDetector(random_state=0)  # unsupervised
   labels = detector.fit_predict(X)
   detector.n_states_, detector.k_louvain_, detector.k_silhouette_  # (4, 6, 3)

**Implementation:** the recipe of the librosa example and of msaf
``scluster``, written for tsseg. The mutual k-NN recurrence and the diagonal
median filter are ported from librosa (ISC), and Louvain from NetworkX
(BSD-3): tsseg depends on neither, and the ports give the same outputs
(checked against librosa 0.11 and NetworkX 3.5). On the nine MoCap trials of
``load_mocap``, the reproduction test recovers the scores of the TSB-SEG
evaluation to the sixth decimal (mean SMS 0.839 guided, 0.822
unsupervised). The paper's own results, on songs, are not reproduced: they
need the audio and its constant-Q features.

**Reference:** McFee & Ellis (2014), *Analyzing song structure with spectral
clustering*, ISMIR; Blondel et al. (2008), *Fast unfolding of communities in
large networks*, J. Stat. Mech. (Louvain).

API reference
-------------

.. automodule:: tsseg.algorithms.laplacian.detector
   :members:
   :show-inheritance:
   :undoc-members:
