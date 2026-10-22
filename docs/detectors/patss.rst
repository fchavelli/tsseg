PaTSS
=====

PaTSS — Pattern-based Time Series Segmentation.

.. warning::
   **Deprecated.** ``PatssDetector`` will be removed in a future release. It
   needs ``npbad``, which is not on PyPI, and it over-segments: on a 54-series
   sample of TSB-SEG it predicted 17 times more change points than the ground
   truth (median), and scored below a random segmentation on univariate series.
   The authors maintain `PaTSEmb <https://github.com/ML-KULeuven/PaTSEmb>`_
   (``pip install patsemb``), which needs a Java runtime.

Description
-----------

PaTSS (Carpentier et al., 2024) describes each time step by the sequential
patterns that occur around it, then clusters these descriptions into states.
For each channel and each window length (the powers of two from ``word_size``
to :math:`\min(2^{10}, n/2)` by default):

1. **Pattern mining** — the sliding windows are discretised with SAX
   (``word_size`` symbols, ``alphabet_size`` bins), and the
   ``top_k_patterns`` most frequent sequential patterns of
   ``min_pattern_size`` to ``max_pattern_size`` symbols are mined (a Java
   miner run by ``npbad``).
2. **Embedding** — each window is described by the relative support of the
   patterns it contains, and each time step by the average over the windows
   that cover it. Patterns whose occurrences overlap those of a kept pattern
   (Jaccard index above ``jaccard_similarity_threshold``) are dropped, and the
   ``nb_largest_variance`` most variable patterns of each channel are kept.
3. **Segmentation** — on the embeddings of all channels and window lengths,
   k-means is run for each number of states in ``n_clusters``, and the
   clustering with the best silhouette score is kept. By default, a logistic
   regression fitted on its labels then gives the probability of each state
   at each time step, which lets two states overlap during a gradual
   transition. The detector returns the most probable state at each time
   step.

All settings are passed through the ``config`` dictionary (see below).

| **Type:** state detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,R\,n^{2})` asymptotically: for each channel and
  each of the :math:`R \le 7` window lengths (16 to 1024 by default),
  ``npbad`` cuts the :math:`n` windows with a mask over the whole series. Add
  the Java pattern miner, whose cost depends on the patterns found, then
  k-means and a silhouette score (quadratic in its sample of at most
  :math:`2000 + 0.1\,(n - 2000)` points) for each of the 8 candidate numbers of
  states. The per-window overhead dominates in practice: about 15 s for 4,000
  univariate points, close to linear in :math:`n` up to 8,000 points

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``config``
     - dict / None
     - ``None``
     - Settings for PaTSS, in three sections (table below). ``None``: every
       default.
   * - ``axis``
     - int
     - ``0``
     - Time axis (input assumed ``(n_timepoints, n_channels)``).

``config`` holds up to three dictionaries, ``"pattern_mining"``,
``"pattern_filter"`` and ``"segmentation"``, and an integer ``"n_jobs"``.
A missing key takes its default value, also inside a section given in part;
the dictionary passed is not modified. Keys are not checked: a misspelt key
is ignored.

.. list-table::
   :header-rows: 1
   :widths: 18 26 16 40

   * - Section
     - Key
     - Default
     - Description
   * - ``pattern_mining``
     - ``resolution``
     - ``-1``
     - Smallest window length, a power of two; ``-1``: ``word_size``.
   * -
     - ``max_resolution_exponent``
     - ``10``
     - Largest window length :math:`2^{10}` (and at most :math:`n/2`).
   * -
     - ``stride``
     - ``1``
     - Step between consecutive windows.
   * -
     - ``word_size``
     - ``16``
     - Number of SAX symbols per window.
   * -
     - ``alphabet_size``
     - ``3``
     - Number of SAX bins.
   * -
     - ``binning_method``
     - ``"global"``
     - SAX bins over the whole min-max normalised series (``"global"``),
       per window (``"local"``) or by k-means (``"kmeans"``).
   * -
     - ``top_k_patterns``
     - ``25``
     - Number of most frequent patterns mined per channel and window length.
   * -
     - ``min_pattern_size``, ``max_pattern_size``
     - ``3``, ``10``
     - Length range of the patterns, in symbols.
   * -
     - ``duration``
     - ``1.2``
     - Constraint on the relative duration of a pattern occurrence.
   * -
     - ``do_mdl``
     - ``False``
     - Filter the mined patterns by minimum description length.
   * - ``pattern_filter``
     - ``min_r_support``, ``max_r_support``
     - ``0.0``, ``1.0``
     - Range of relative support of the patterns kept.
   * -
     - ``jaccard_similarity_threshold``
     - ``0.9``
     - A pattern whose occurrences overlap those of a kept pattern beyond
       this Jaccard index is dropped.
   * -
     - ``nb_largest_variance``
     - ``50``
     - Number of patterns kept per channel, the most variable ones.
   * -
     - ``do_pca``
     - ``False``
     - Project the embedding onto ``nb_largest_variance`` principal
       components instead of selecting patterns.
   * - ``segmentation``
     - ``algorithm``
     - ``"logistic_regression"``
     - ``"logistic_regression"`` (k-means, then a logistic regression on its
       labels gives the state probabilities), ``"gmm"`` (Gaussian mixture
       and its probabilities) or ``"k_means"`` (k-means labels only).
   * -
     - ``regularization``
     - ``0.1``
     - Inverse regularisation strength ``C`` of the logistic regression;
       ``reg_covar`` of the Gaussian mixture.
   * -
     - ``n_clusters``
     - ``[2, ..., 9]``
     - Candidate numbers of states; the best silhouette score wins.
   * - (top level)
     - ``n_jobs``
     - ``1``
     - Number of processes for the pattern mining and the clustering.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import PatssDetector

   detector = PatssDetector()       # default config
   states = detector.fit_predict(X)

   # Custom config: missing keys keep their default value
   detector = PatssDetector(config={
       "pattern_mining": {"word_size": 8, "top_k_patterns": 50},
       "segmentation": {"n_clusters": [2, 3, 4]},
   })
   states = detector.fit_predict(X)

**Implementation:** Adapted from the original PaTSS code (KU Leuven, MIT
licence).

**Reference:** Carpentier, L., Feremans, L., Meert, W. & Verbeke, M. (2024).
*Pattern-based Time Series Semantic Segmentation with Gradual State
Transitions.* Proceedings of the 2024 SIAM International Conference on Data
Mining (SDM), 316–324. `doi:10.1137/1.9781611978032.36
<https://doi.org/10.1137/1.9781611978032.36>`_

API reference
-------------

.. automodule:: tsseg.algorithms.patss.detector
   :members:
   :show-inheritance:
   :undoc-members:
