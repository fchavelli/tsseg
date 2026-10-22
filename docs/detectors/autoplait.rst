AutoPlait
=========

AutoPlait — parameter-free mining of co-evolving time sequences.

Description
-----------

AutoPlait describes a (multivariate) time series as segments, each generated
by one of a few *regimes*. A regime is a hidden Markov model with diagonal
Gaussian emissions; segments of the same regime share it. The number of
regimes, the number of hidden states of each model and the cut points are
chosen by minimising the total description length (MDL) of the models and of
the data, without any user parameter.

The search is top-down. Starting from a single regime that covers the whole
series, AutoPlait tries to split each regime in two:

1. **Seeds** — pairs of windows of length :math:`\lceil n/10 \rceil` taken
   uniformly in the regime each initialise an HMM; the pair whose split is the
   shortest to describe is kept.
2. **CutPointSearch** — a Viterbi pass over the two HMMs, with a switching
   probability per regime, assigns each part of the regime to one of them;
   segments that the other model describes almost as well are moved to it.
3. **RegimeSplit** — model estimation (Baum-Welch, number of states chosen by
   MDL) and cut point search alternate until the description length stops
   decreasing.

The split is kept when it shortens the description of the regime by more than
3 %; the two new regimes are then split in turn.

| **Type:** state detection
| **Supervision:** unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n)` per cut point search (for a bounded number of
  states and of seeds)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``n_cps``
     - int / None
     - ``None``
     - Deprecated and ignored. AutoPlait chooses the number of segments
       itself and has no guided mode, so this value has never changed the
       output. Passing it emits a ``FutureWarning`` at ``fit``; it will be
       removed in a future release.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import AutoPlaitDetector

   detector = AutoPlaitDetector()
   labels = detector.fit_predict(X)

``labels`` holds, for each time point, the number of the AutoPlait regime its
segment belongs to: segments of the same regime share a label.

.. note::
   AutoPlait needs numba (``pip install "tsseg[accelerators]"``).

   The C code floors the log-density of an observation, summed over its
   channels, at :math:`\log 10^{-6}`, and the translation keeps this floor.
   With many channels, the observations reach it under every model and the
   models can no longer be told apart. About 20 % of the time points of a HAS
   series (9 channels) and of ActRecTut subject 1 (10 channels) reach it under
   their own regime, and every time point of a 20-channel Gaussian series with
   a 3-sigma mean shift, on which AutoPlait finds a single regime.

**Implementation:** Python translation, compiled with numba, of the C code of
Matsubara et al. (version of June 18, 2014), random draws included: it returns
the segmentation of the C binary built with GCC on Linux. Each channel is
z-normalised in float32, as the C code reads its input. Unlike the binary, the
translation has no limit on the number of channels, of regimes or of segments,
and sets a constant channel to zero.

**Reference:** Matsubara, Sakurai & Faloutsos (2014), *AutoPlait: Automatic Mining
of Co-evolving Time Sequences*, SIGMOD.

API reference
-------------

.. automodule:: tsseg.algorithms.autoplait.detector
   :members:
   :show-inheritance:
   :undoc-members:
