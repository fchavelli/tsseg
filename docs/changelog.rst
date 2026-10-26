:orphan:

.. _changelog:

Changelog
=========

This changelog highlights notable updates. For full commit history, refer to
`GitHub <https://github.com/fchavelli/tsseg/commits/main>`_.

Unreleased
----------

* ``Time2StateDetector`` and ``E2USDDetector`` refuse in ``fit``, with a
  ``ValueError``, the parameters their encoder cannot run: an odd
  ``window_size`` (accepted by ``fit``, it made ``predict`` fail after the
  training), ``kernel_size=1`` for Time2State and a ``kernel_size`` above
  ``min(window_size, 64)`` for E2USD (both failed in ``fit`` with a
  ``RuntimeError`` from PyTorch).
* ``Time2StateDetector`` and ``E2USDDetector`` label a constant series as
  one state. When the variance of the series was exactly zero, as for
  ``np.zeros(n)``, the z-normalisation divided by zero and ``predict``
  raised ``ValueError: Input X contains NaN``; a constant series whose
  variance is a rounding error, as for ``np.full(n, 0.1)``, already got one
  state. A series of zero variance is now only centred. The other series
  are normalised as before.
* ``TGLADDetector`` now trains uGLAD for 2000 epochs per batch by default,
  as the authors' code does, instead of 100. After 100 epochs, the soft
  threshold that GLAD learns could set every partial correlation to zero, and
  the detector then found no change point, as on a two-channel series with
  two clear mean shifts and on the first three MoCap trials with the default
  ``window_size``; after 2000 epochs it finds 21, 37 and 34 change points on
  these trials, for 6, 9 and 8 true ones. Training takes about 20 times
  longer; pass ``epochs=100`` for the previous behaviour. The documentation
  now also says that the windows around one transition can each give a
  change point, and that on a univariate series the summary of a window, the
  precision of its single channel (a fallback that is not in the authors'
  code), does not follow the variance of the signal, as the docstring said:
  uGLAD min-max scales each window first.
* ``TGLADDetector`` prints nothing with ``verbose=False``, its default. The
  vendored uGLAD solver printed three lines on stdout for each batch of
  windows ("Running uGLAD in multi-task mode", "Processing the input table
  for basic compatibility check", "Total runtime: ...") whatever
  ``verbose`` said; with ``verbose=True`` they are still printed, with the
  loss every tenth of the epochs. ``epochs`` below 10, which the parameter
  schema accepts, raised ``ZeroDivisionError`` at ``fit``: the solver
  printed the loss every ``int(epochs / 10)`` epochs, which is 0 below 10
  epochs; it now prints it every epoch there.
* ``clone(random_state=k)`` seeds the clone. The seed was ignored: ``clone``
  only called a ``set_random_state`` method that no detector defines, so the
  clone kept the ``random_state`` of the original. As in aeon, the
  ``random_state`` parameter of the clone is now set to an integer drawn from
  ``check_random_state(k)``, the same for the same ``k``; ``k`` may also be a
  ``numpy.random.RandomState``. ``clone()`` without a seed, and detectors
  without a ``random_state`` parameter, are unchanged.
* ``clone()`` copies the parameters, as scikit-learn's ``clone`` (used by
  aeon) does. The clone received the parameter objects themselves, so a
  mutable parameter (a list such as ``ICIDDetector(psi_list=...)``, a dict, an
  array, a ``RandomState``) was shared: changing it in place, or drawing from
  the generator, also changed the original. Estimators passed as parameters,
  also inside a dict, list or tuple, are now cloned unfitted, and
  ``clone(random_state=k)`` also seeds their ``random_state``, in aeon's
  order. A constructor that converts a parameter is still accepted, unlike
  scikit-learn's ``clone``.
* ``Time2StateDetector`` and ``E2USDDetector`` change their output at the
  end of the series. Each window votes for its state on the points it
  covers, and the last ``(n_timepoints - window_size) % step`` points, which
  no window covers, got state 0 whatever the state around them, as in the
  code both detectors wrap: 16 of 630 points with ``window_size=64``
  and ``step=50``. These points now take the state of the last window. The
  other points keep their state.
* Licences: ``Time2StateDetector`` and ``E2USDDetector`` ship code from
  projects that their licence files did not name. The causal CNN encoder and
  the ``Dataset`` class of ``time2state.py`` come from the code of Franceschi,
  Dieuleveut and Jaggi (2019), under the Apache License 2.0, and partly from
  locuslab/TCN, under the MIT License; ``e2usd.py`` holds the same
  ``Dataset`` class and the ``moving_avg`` and ``series_decomp`` blocks of
  Autoformer, under the MIT License. The ``LICENSE`` file of
  ``tsseg/algorithms/time2state`` now holds these licences, the ``NOTICE`` of
  Franceschi et al. and the changes made to their code, and a new ``LICENSE``
  file in ``tsseg/algorithms/e2usd`` does the same for the two parts of
  E2USD (the rest of E2USD still has no published licence). The code is
  unchanged.
* Licences: ``GreedyGaussianDetector`` is adapted from aeon
  (``aeon.segmentation._ggs``), whose code is adapted from cvxgrp/GGS, but
  ``tsseg/algorithms/ggs/LICENSE`` held only the BSD 2-Clause licence of
  cvxgrp/GGS, and the README of the directory and of tsseg named only
  cvxgrp/GGS. The ``LICENSE`` file now holds aeon's BSD 3-Clause licence too,
  and the READMEs, the detector page and the module docstring name both
  sources. The code is unchanged.

* Docs: the Hidalgo page, README and docstring say what the detector does not
  separate. It tells states apart by their intrinsic dimension only, from
  distance ratios that location and scale leave unchanged, without the time
  order: states that differ in mean or variance but have the same dimension
  are not separated. Spatially separated regions of the same dimension are
  distinct in the model, but the Gibbs chain is slow to split them; the paper
  runs :math:`10^5` iterations from 10 random starts, the detector 1000 from
  one by default. The detector is unchanged.
* The documentation of ``TiccDetector`` now describes what the detector
  computes: the window of time point ``t`` holds the points ``t`` to
  ``t + window_size - 1``, the last ``window_size - 1`` observations enter no
  window, ``threshold`` has no effect, ``beta`` is twice the :math:`\beta` of
  the paper, ``predict`` returns the labels of the series given to ``fit``
  whatever its input. Every public member of ``tsseg.algorithms.ticc.ticc``
  has a NumPy docstring. The detector is unchanged.
* ``TiccDetector`` with ``n_states`` given raises a ``ValueError`` that names
  the cause when a state cannot get the two windows it needs to estimate its
  covariance: when the series has fewer than ``2 * n_states`` windows
  (``n_timepoints - window_size + 1``), or when the initial Gaussian mixture
  gives a state fewer than two windows, as it can to a few outlying windows.
  ``fit`` raised a bare ``KeyError`` such as ``KeyError: (2, 1)``, or the
  ``ValueError`` of scikit-learn's Gaussian mixture when the series had fewer
  windows than states. With ``n_states=None``, such a number of states still
  gets an infinite BIC; labels are unchanged.
* The documentation of ``Time2StateDetector`` now describes what the
  detector computes and where it fails: ``batch_size`` has no effect,
  ``window_size`` must be even, ``kernel_size`` must be at least 2,
  ``random_state`` does not seed the clustering of ``predict``, and a
  constant series makes ``predict`` raise. Every public member of
  ``tsseg.algorithms.time2state.time2state`` has a docstring; wrong ones are
  corrected (the ``final`` flag of ``CausalConvolutionBlock`` adds the last
  activation, ``reduced_size`` of ``CausalCNNEncoder`` is a number of
  channels, the encoder outputs two-dimensional tensors). The README of the
  detector attributed the method to the wrong authors and venue; it now
  cites Wang, Wu, Zhou and Cai, *Proc. ACM Manag. Data* 1(1), 2023.
* The documentation of ``E2USDDetector`` now describes what the detector
  computes and where it fails: ``batch_size`` and ``depth`` have no effect,
  ``window_size`` must be even, ``kernel_size`` can be at most
  ``min(window_size, 64)``, the convolutions of the encoder keep their random
  initial weights, ``random_state`` does not seed the clustering of
  ``predict``, a constant series makes ``predict`` raise, and ``y`` is
  ignored (the docstring of ``fit`` said it set the number of states). Every
  public member of ``tsseg.algorithms.e2usd.e2usd`` has a docstring. The
  README of the detector credited "Chen, Zhichen and others"; it and the
  documentation page now cite Lai, Li, Zhang, Zhao, Qian and Jensen, WWW
  2024, pp. 3010-3021.
* The documentation of ``BeastDetector`` now says that a fixed
  ``mcmc_seed`` reproduces the output on one machine only: Rbeast picks its
  numerical routines by CPU instruction set, and their rounding differences
  change the MCMC samples. It also says that, on some machines, Rbeast
  returned NaN for every output on a 6,000-point series, so that the detector
  returned no change point. The test of that series, which failed on such
  machines, now expects this failure on CPUs that support AVX-512.
* ``BeastDetector`` reruns Rbeast once with its SSE routines
  (``cputype="sse"``) when the median number of change points
  (``ncp_median``) of a run is NaN in every component and every channel, and
  raises a ``RuntimeWarning``. On some machines, Rbeast returned NaN for every
  output on a 6,000-point series, and the detector returned no change point.
  Other runs are unchanged; a failed run costs a second, slower one (1.5 to
  2 times as long on an AVX2 CPU). If the rerun returns NaN too, the
  detector still returns no change point.
* ``GreedyGaussianDetector`` no longer depends on the level of a
  multivariate series. Its split search started the running mean of the
  left part from the first sample divided by the number of channels, a line
  taken from the reference implementation of GGS, so every candidate split
  was scored with a wrong left covariance: on three segments of two
  channels, ``k_max=2`` found ``[150, 250]``, and only ``[250]`` once 10 was
  added to the series. Each split is now scored by the regularised
  log-likelihood of the paper, the one ``log_likelihood`` computes.
  Multivariate outputs change; univariate outputs, where the two starts
  agree, do not.
* ``GreedyGaussianDetector(lamb=0)`` is refused by ``fit`` on a multivariate
  series. Without regularisation, the covariance of a segment with no more
  points than channels is singular, and the split search scores segments of
  two points: ``predict`` raised ``LinAlgError: Matrix is not positive
  definite`` on every multivariate series tried (120 of 120 with two to four
  channels and the default ``k_max``). The paper assumes
  :math:`\lambda > 0`. On a univariate series, ``lamb=0`` still runs; if
  repeated values make a segment's variance zero, the ``LinAlgError`` now
  says to use ``lamb > 0``.
* Every detector can be imported without its optional dependencies, so
  ``from tsseg.algorithms import *`` and ``hasattr(tsseg.algorithms, name)``
  work after a bare ``pip install tsseg``. Seven detectors (``ClapDetector``,
  ``ClaspDetector``, ``EAggloDetector``, ``E2USDDetector``,
  ``Time2StateDetector``, ``TireDetector``, ``ProphetDetector``) raised
  ``ModuleNotFoundError: No module named 'numba'`` (or ``'torch'``,
  ``'prophet'``) on import, which made the star import fail. The
  dependencies listed in a detector's ``python_dependencies`` tag are now
  checked once, when it is constructed, as in aeon; a missing one raises
  ``ModuleNotFoundError`` with the extra to install, the same way for all
  fourteen detectors with optional dependencies, e.g. ``ClapDetector
  requires numba, aeon: pip install "tsseg[accelerators,aeon]"``. Before,
  some failed only at ``fit`` (``AutoPlaitDetector``, ``SNLDSDetector``,
  ``BeastDetector``, ``ChangeForestDetector``) and some without naming the
  extra. Accessing ``tsseg.algorithms.TireDetector`` (or E2USD, Time2State)
  no longer imports torch, and ``import tsseg.algorithms`` no longer imports
  stumpy: they are imported when a detector that needs them is constructed
  or fitted. Outputs do not change.
* The wheel and the source distribution now include the license of the
  PaTSS code, ``tsseg/algorithms/patss/LICENCE``, which the package-data
  patterns (``**/LICENSE``) left out. MATLAB data files (``*.mat``) are no
  longer selected as package data; none is left in the package.
* New extra ``tsseg[fluss]``, which installs stumpy for ``FLUSSDetector``;
  ``tsseg[all]`` now includes it. stumpy was declared nowhere, so
  ``pip install "tsseg[all]"`` left ``FLUSSDetector`` unusable. The CI
  installs the extra, so the tests of ``FLUSSDetector``, which it skipped,
  now run there; the extras table of the documentation lists it.
* tsseg no longer depends on ``future``, which nothing imports, and the
  ``dev`` extra no longer installs ``black``: the code is formatted with
  ``ruff format``.
* ``tsseg/algorithms/base.py``, whose base classes are adapted from aeon, now
  opens with aeon's BSD 3-Clause license, which asks redistributed source
  code to keep it. The README said that this license covered the file, but
  neither the file nor its directory held a copy.
* ``ChangeFinderDetector(n_cps=k)`` returns ``k`` change points. When fewer
  than ``k`` peaks of the score passed the threshold, it returned only those
  (``[306]`` for ``n_cps=2`` on a series with two changes): the step meant to
  relax the threshold went over the same thresholded peaks and added none.
  The local maxima below the threshold now complete the selection, by
  decreasing score, ``min_distance`` apart; fewer than ``k`` change points
  remain possible only when the score has fewer such maxima. Guided outputs
  change when the threshold kept fewer than ``k`` peaks, including per
  channel with ``multivariate_strategy="ensembling"``; outputs without
  ``n_cps`` do not change.
* New ``SeededBinSegDetector``: seeded binary segmentation (Kovács, Bühlmann,
  Li and Munk, 2023), written from the paper. Best split of every interval of
  a deterministic, near-linear collection, then greedy or
  narrowest-over-threshold selection; number of change points from
  ``n_cps``, a ``penalty`` on the gain, or the strengthened Schwarz
  information criterion. ``intervals="wild"`` gives wild binary segmentation
  (Fryzlewicz, 2014). Any ruptures cost; ``"l2"`` is the CUSUM of the papers.
  For a multivariate series, ``"l2"`` with the criterion or ``n_cps`` first
  divides each channel by its noise level (median absolute deviation of its
  differences), so that the result does not depend on the scale of each
  channel. ``augment=True`` adds the optional augmentation of wild binary
  segmentation to the greedy selection. With ``intervals="wild"`` and no
  ``random_state``, the detector is tagged ``non_deterministic``.
* Add a ``CITATION.cff`` file, read by GitHub's *Cite this repository*
  button, and the citation of the TSB-SEG benchmark (NeurIPS 2026) to the
  README.
* ``HidalgoDetector`` takes its arguments by keyword only and has a last
  parameter ``axis`` (default ``0``): ``HidalgoDetector(axis=1)`` reads series
  of shape ``(n_channels, n_timepoints)``. A positional call such as ``HidalgoDetector("euclidean")``
  now raises a ``TypeError``; ``fit(X, axis=1)`` works as before.
* Installation from source no longer needs conda. ``make install`` creates a
  virtual environment in ``.venv`` with the ``venv`` module and installs tsseg
  in it, editable, with the ``all`` and ``dev`` extras (options ``EXTRAS``,
  ``VENV``, ``VENV_PYTHON``, and ``TORCH_INDEX_URL`` to take torch from
  another index, such as its CPU build). The conda environment ``tsseg-env``
  is now created by ``make install-conda`` (which installs ``EXTRAS`` too,
  hence ``dev`` besides ``all``) and removed by ``make clean-conda``;
  ``make clean`` removes ``.venv`` only. The README, the
  documentation home page and the getting started and contributing guides
  give the ``venv`` + ``pip`` steps first, and say that ``tsseg[all]`` needs
  Python 3.10 to 3.12 (the TensorFlow versions pinned by ``tsseg[tscp2]`` have
  no wheel for 3.13 or later).
* ``EspressoDetector(random_state=...)`` seeds the matrix profile: the seed was
  stored but never passed on, so every call drew a fresh random order of the
  subsequences. That order decides which nearest neighbour is kept when
  distances tie, which happens on exactly repeated patterns; there, the
  change points changed from one run to the next whatever the seed. A fixed
  seed now gives the same change points at every run.
* Docs: the ESPRESSO page described the method wrongly. It said that the
  matrix profile is computed with ``stumpy`` (tsseg computes it in NumPy and
  never imports ``stumpy`` for ESPRESSO) and that the peaks of the density
  curve are the change points (its minima are candidates, among which a greedy
  information-gain search picks ``n_segments - 1``), and called the code new
  while it translates the authors' MATLAB code. It now describes the four
  steps, the bounds of ``window_size`` and the published reference
  (Proc. ACM IMWUT).
* ``ClapDetector`` accepts the classifiers and merge scores that CLaP
  implements: ``classifier`` in {``"rocket"``, ``"mrhydra"``, ``"weasel"``,
  ``"quant"``, ``"rdst"``, ``"proximityforest"``, ``"freshprince"``,
  ``"inception"``} and ``merge_score`` in {``"cgain"``, ``"f1_score"``,
  ``"log_loss"``, ``"ami"``, ``"hamming_loss"``, ``"roc_auc"``}. Since 0.1.1 its
  parameter check refused all of them but ``"rocket"`` and ``"cgain"``, so
  ``classifier="mrhydra"`` (the Fixed configuration of CLaP in TSB-SEG) raised
  ``ValueError``; it allowed ``"knn"``, ``"svm"`` and ``"score"``, which failed
  later, in ``fit``, and are now refused upfront.
* ``ClapDetector(change_points=...)`` (known change points) works: an array was
  refused by the parameter check and a list failed in ``fit``
  (``AttributeError``). Lists, tuples and arrays are accepted.
* ``ClaspDetector`` accepts ``score="f1"``, which ClaSP implements and
  documents; the parameter check refused it.
* ``ClaspDetector`` and ``ClapDetector`` in guided mode (``n_change_points``,
  or an integer ``n_segments``) return the number of change points they are
  given, fewer only when the segments are too short to split further. ClaSP
  validated each candidate in this mode too, as claspy does, so a count could
  give fewer change points, or none: on a series of mean shifts in noise,
  ``n_change_points=2`` gave no change point and ``ClapDetector`` a single
  state. The validation now applies only when the number is learned
  (``n_segments="learn"``, the default), and ``validation`` and
  ``threshold`` of ``ClaspDetector`` have no effect in guided mode. Guided
  results change; unsupervised results do not.
* ``ClapDetector`` has a ``distance`` parameter, passed to ClaSP:
  ``"znormed_euclidean_distance"`` (default, unchanged),
  ``"euclidean_distance"`` or ``"cinvariant_euclidean_distance"``, as in
  ``ClaspDetector``. The z-normalised default compares the shapes of the
  windows and misses a change of level alone, such as a mean shift in noise.
* Docs: ``ClapDetector(random_state=...)`` seeds the classifiers of CLaP only;
  ClaSP keeps the fixed seed of claspy (2357).
* Guided mode on a series without any change point: ``TSCP2Detector(n_cps=0)``,
  ``BOCDDetector(max_cps=0)``, ``BeastDetector(max_cps=0)`` and
  ``GreedyGaussianDetector(k_max=0)`` return no change point; their parameter
  check refused ``0`` (``ValueError``). ``BOCDDetector`` also checked
  ``max_cps`` only after selecting a change point, so ``max_cps=0`` alone would
  still have returned one on univariate series and with the ``"l2"`` strategy.
* ``FLUSSDetector`` with ``n_segments`` set returns its change points sorted,
  as the change point detection output requires. On one channel or with
  ``multivariate_strategy="l2"`` and three segments or more, they came in the
  order in which ``stumpy.fluss`` extracts them (lowest corrected arc curve
  first), for example ``[3278, 1600]`` on MoCap trial 0. The change points and
  their scores are unchanged, only their order.
* ``FLUSSDetector`` with ``n_segments`` set no longer returns ``0`` as a change
  point. Once the corrected arc curve left after the exclusion zones of the
  change points found is flat (1 everywhere), ``stumpy.fluss`` takes its
  minimum at index 0 for each change point still to find: ``[0, 681]`` on 1000
  points of white noise (``window_size=50``, three segments), ``0`` repeated
  when ``n_segments`` asks for more change points than the exclusion zones
  leave room for. These are dropped, so fewer than ``n_segments - 1`` change
  points can come out; on one channel or with ``multivariate_strategy="l2"``,
  the others are unchanged. With ``"ensembling"``, the channels no longer vote
  for ``0``, and the change points they agree on can change.
  ``n_segments=1`` returns an empty array instead of an empty list.
* ``FLUSSDetector`` has a new parameter ``normalize`` (default ``True``,
  passed to ``stumpy.stump``). The default keeps the z-normalized matrix
  profile of the FLUSS paper, to which a change in mean or scale alone is
  invisible; ``normalize=False`` compares the raw subsequences, so shifts in
  level show (``[293, 692]`` instead of ``[485, 739]`` for change points at
  300 and 700 of white noise with three means). Without ``n_segments``, it
  needs a lower ``threshold`` than the default 0.5. Outputs with the default
  are unchanged.
* ``BinSegDetector`` and ``BottomUpDetector``: ``penalty`` defaults to
  ``None``, which means 10 when neither ``n_cps`` nor ``epsilon`` is given.
  With the former default of 10, ``n_cps=k`` alone always warned that the
  penalty was ignored, and ``epsilon`` alone raised ``ValueError`` (it needed
  ``penalty=None``). The change points are the same in every case that was
  accepted before; ``penalty=None`` alone now means a penalty of 10 instead of
  raising ``ValueError``.
* Docs: the BinSeg page listed an ``"ar"`` cost model, which the parameter
  check refuses, and omitted ``"cosine"``, which it accepts.
* Docs: the BottomUp and vendored ruptures pages state that ``BottomUp`` has
  the speed-up of ruptures v1.1.9 (PR #309), which tsseg has shipped since its
  first release; they said v1.1.8 only. On 100,000 points, the Python backend
  segments in 2 s instead of 15 s for the v1.1.8 code, with the same change
  points.
* Docs: the docstrings of ``ClaspDetector``, ``EspressoDetector``,
  ``FLUSSDetector``, ``HMMDetector``, ``InformationGainDetector``,
  ``RandomDetector``, ``TireDetector`` and ``TSCP2Detector`` document every
  constructor parameter. ``EspressoDetector`` documented ``subsequence_length``
  instead of ``window_size``, and ``InformationGainDetector`` a
  ``change_points_`` attribute that is never set. The FLUSS, HMM and IGTS
  examples use tsseg instead of aeon. Now documented: ``RandomDetector``
  ignores ``n_change_points`` and ``n_states`` unless ``semi_supervised=True``.
* ``TireDetector`` with ``stride > 1`` returns time point indices. It returned
  window indices (with ``stride=4``, the three changes of a 600-point series
  at 150, 300 and 450 came out at 42, 76 and 113), and ``change_scores_`` had
  one value per window instead of one per time point, so
  ``peak_distance_fraction`` was counted in windows. The scores are now placed
  midway between the two compared windows and interpolated to every time
  point; ``stride=1``, the default, gives the same output as before.
* ``TireDetector(n_segments=1)`` returns no change point; it selected the
  peaks by ``prominence_threshold``, as without ``n_segments``.
* Docs: the contributing guide names the six modules of the generic detector
  test suite (it listed four), states the change-point output they check
  (strictly increasing indices in ``[1, N - 1]``) and the extras the CI
  installs; ``tests/README.md`` describes the current helpers of
  ``tests/algorithms/conftest.py``.

* ``AutoPlaitDetector`` no longer needs a C binary: the C code of Matsubara et
  al. (version of June 18, 2014) is translated to Python and compiled with
  numba (``tsseg[accelerators]``), random draws included. It returns the
  segmentation of the binary built with GCC on Linux (same segments, same
  total cost on the 9 MoCap trials of ``load_mocap``, the 7 demo series of the
  C code and 38 series of ActRecTut, HAS, PUMP, SKAB, TSSB, USC-HAD and UTSA)
  and runs 2 to 8 times faster. Series of more than 15 channels, which the binary read as a
  longer series (``ValueError`` since 0.3.0), are segmented; a constant
  channel is set to zero instead of yielding ``NaN`` values. ``c/autoplait/``,
  the runner ``tsseg.algorithms.autoplait.autoplait_c`` and the
  ``AUTOPLAIT_DIR`` variable are removed. ``AutoPlaitDetector`` gains the
  ``n_regimes_`` and ``cost_`` attributes.

* ``TiccDetector`` runs without ``n_states``: with ``n_states=None``, the new
  default (it was ``5``), the number of states is chosen by the Bayesian
  information criterion among ``1, ..., max_states`` (new parameter, default
  ``8``), as the TICC paper recommends. The criterion counts the free entries
  of each block Toeplitz precision matrix, the means and the switches, and
  divides the log-likelihood of the overlapping windows by ``window_size``
  (see the docstring); the attributes ``n_states_`` and ``bic_`` report the
  choice. The detector is tagged ``capability:unsupervised``. Pass
  ``n_states=5`` to keep the former behaviour.
* The TICC solver returns the same labels two to five times faster: it runs in
  the calling process when ``num_proc=1`` (it always started a process pool),
  its ADMM Z-update and its assignment step are vectorised, and it no longer
  computes, at each iteration, a k-means clustering and confusion matrices
  that it did not use. It no longer reseeds NumPy's global random generator
  nor changes the print options of NumPy and pandas.
* ``EAggloDetector.predict(X)`` segments ``X`` when it is not the series given
  to ``fit``. It compared the pandas index only, so on another series of the
  same length it returned the change points of the fitted series; a shorter
  series was completed with the end of the fitted series before being
  segmented. A new series is segmented alone, with a warning as before, and the
  fitted attributes are left unchanged; ``fit_predict`` and ``predict`` on the
  fitted series are unchanged. With ``member``, a series of another length
  raises a ``ValueError`` that says so (a longer one failed in pandas, a
  shorter one was completed).
* ``HidalgoDetector.predict(X)`` labels ``X`` when it is not the series given
  to ``fit``. It returned the labels of the fitted series whatever ``X``, so as
  many labels as the fitted series has time points. A new series is labelled
  with a new fit, with a warning, and the fitted attributes are left
  unchanged; ``fit_predict`` and ``predict`` on the fitted series are
  unchanged. The docstring now says that time points whose most probable state
  has a posterior probability below 0.8 are labelled ``-1``.
* The pages of IGTS, TS-CP2, PaTSS and SNLDS give their time complexity, so
  every detector page now does. The SNLDS page said that a training step
  costs :math:`O(B\,L\,K^{2})`; :math:`K^{2}` is only the forward-backward
  recursion, and the discrete transition network costs :math:`O(K^{4})` per
  time step.
* ``E2USDDetector.predict`` and ``fit_predict`` write nothing to the standard
  output. Since 0.1.0 they printed the shape of the array of windows, for
  example ``(17, 1, 32)``, on every call.
* ``TSCP2Detector`` leaves ``window_size`` unchanged on a short series. On a
  series of exactly ``2 * window_size`` points, ``fit`` halved
  ``window_size`` in place and printed a warning to the standard output:
  ``get_params`` returned the halved value, and a later ``fit`` of the same
  instance on a longer series kept it. ``fit`` now trains on the halved
  window with a ``UserWarning``; its change points are unchanged.
  ``predict`` on a series shorter than two windows of ``fit`` raises a
  ``ValueError`` that says so: it failed inside the encoder and left the
  detector unusable for every later ``predict``. ``predict`` on a series of
  exactly two windows scores its single window pair instead of failing.
* ``PatssDetector(config=...)`` leaves the dictionary it is given unchanged:
  ``predict`` completed it in place with every missing default, so
  ``get_params()["config"]`` changed after the first ``predict``, and the
  sections it added were the module-level default dictionaries themselves
  (editing them changed the defaults of every later run).
* ``PatssDetector`` with ``config={"segmentation": {"algorithm": "k_means"}}``
  returns one state label per time point; it returned a single integer.
* The PaTSS page lists the keys that ``config`` takes, with their defaults:
  the keys it cited (``pattern_length``, ``min_frequency``, ``max_overlap``,
  ``n_segments``) do not exist, so its custom configuration example had no
  effect. It describes the three steps as the code runs them and gives the
  reference of the paper (SDM 2024).
* ``PatssDetector`` writes nothing to the standard output; it printed the
  temporary directory of every window size it tried.
* ``BeastDetector`` no longer fails with ``ValueError: cannot convert float
  NaN to integer`` when Rbeast returns NaN for the median number of change
  points (``ncp_median``), as it did in CI on a 6,000-point series. It warns
  (``RuntimeWarning``) and keeps the change points above
  ``cp_prob_threshold`` without capping their number; if every Rbeast output
  is NaN, it returns no change point.
* The iCID, tGLAD and TIRE pages said "*Origin: new code.*" They now say where
  the code comes from: a translation of the authors' MATLAB code (iCID,
  GPLv3), a wrapper around the uGLAD solver vendored from the authors'
  repository under a non-commercial licence (tGLAD), and a PyTorch
  reimplementation of the authors' TensorFlow code (TIRE). The ``threshold``
  of ``TGLADDetector`` applies to the absolute second difference of the sum of
  partial correlations, not to a Frobenius distance as the page, the README
  and the docstring said. The iCID page and README give the right title and
  authors of the JAIR paper, and the tGLAD README cites the paper of its
  authors.
* Docs: the Prophet page, README and docstring said that a level shift shows
  as two opposite rate changes on consecutive candidates. Prophet fits it as a
  steep ramp, a run of rate changes of one sign before the shift and a run of
  the opposite sign after it, which can be wider than ``tolerance``: one shift
  can then give two change points and leave a smaller shift out of the
  ``n_changepoints`` returned. The detector is unchanged.
* ``SNLDSDetector`` seeds its initial weights with ``random_state``. Keras 3
  draws the seed of an initializer created without one from Python's
  ``random`` module, which the detector did not seed: two fits with the same
  ``random_state`` started from different weights, and ``fit`` consumed the
  caller's ``random`` stream. On a series of three segments with clear mean
  shifts, two runs with the default parameters gave adjusted Rand indices of
  0.52 and 0.92. Training still differs from run to run, even with
  ``tf.config.experimental.enable_op_determinism()``: the documentation now
  says so, with what it does to the result on that series, where it said
  that only a few labels could differ between two calls.

0.3.0 (2026-10-14)
------------------

Upgrading from 0.2.0: ``VQTSSDetector``, ``VSAXDetector`` and the HDP-HSMM
aliases are removed; ``GreedyGaussianDetector`` returns change points instead
of labels; the arguments of ``SNLDSDetector`` are keyword-only;
``ProphetDetector``, and ``ChangeFinderDetector`` without ``n_cps``, select
change points differently by default; ``BeastDetector`` needs ``Rbeast`` from
PyPI (``tsseg[beast]``); the supervision and dependency tags of several
detectors change. ``KCPDDetector``, ``HMMDetector``, ``PatssDetector`` and the
``n_cps`` parameter of ``AutoPlaitDetector`` are deprecated. Several fixes
change the output of AutoPlait, BEAST, IGTS, Window and the vendored ruptures
solvers (ties): results computed with 0.2.0 can differ.

* New detector ``ChangeForestDetector``: classifier-based nonparametric change
  point detection with random forests (Londschien, Bühlmann and Kovács,
  JMLR 2023). It wraps the official ``changeforest`` package, installed with
  the new extra ``pip install tsseg[changeforest]``. With ``n_cps`` it runs
  best-first binary segmentation on the package's gains, a semi-supervised
  mode that is not part of the paper.
* New ``RuLSIFDetector`` (Liu, Yamada, Collier and Sugiyama, Neural Networks
  2013): change points are peaks of the symmetrised relative Pearson
  divergence between the subsequences before and after each point, estimated
  by RuLSIF with cross-validated kernel width and regularisation. NumPy port
  of the authors' MATLAB code (MIT); guided (``n_cps``) or thresholded.
* ``TSCP2Detector`` takes a ``random_state`` seed (``None`` by default, as
  before): with an integer, every fit initialises the encoder, shuffles the
  training windows and draws the dropout masks identically, so two fits on the
  same series return the same change points. Without a seed, the weights and
  the shuffling changed at every fit, and nothing declared the detector
  non-deterministic. The seed goes through ``tf.keras.utils.set_random_seed``,
  which also reseeds the global Python, NumPy and TensorFlow generators.
* Dependencies: ``tqdm`` and ``joblib`` are no longer required. No module of
  tsseg imports them since the unfinished Python port of AutoPlait was removed
  (``joblib`` remains installed with scikit-learn). ``fetch_dataset`` shows a
  download progress bar only when ``tqdm`` is installed.

* Tags: the undocumented ``semi_supervised`` tag of ``AmocDetector``,
  ``EspressoDetector`` and ``RandomDetector`` is removed:
  ``capability:semi_supervised`` is the tag (the ``semi_supervised`` parameter
  of ``RandomDetector`` is unchanged). The deprecated ``HMMDetector``, neither
  unsupervised nor semi-supervised, declares ``capability:semi_supervised``:
  its number of states is the length of ``emission_funcs``. A contract test
  rejects unknown tags and checks that every detector declares both
  supervision tags, at least one of them ``True``.

* Tags: ``python_dependencies`` is ``None`` or a list of the optional modules
  a detector imports. ``ProphetDetector``, ``TGLADDetector`` and
  ``TireDetector`` declared a string (``"torch,networkx"``), and eight
  detectors declared nothing; they now declare ``["numba", "aeon"]``
  (``ClapDetector``), ``["numba"]`` (``ClaspDetector``, ``EAggloDetector``),
  ``["torch"]`` (``E2USDDetector``, ``Time2StateDetector``), ``["stumpy"]``
  (``FLUSSDetector``), ``["npbad"]`` (``PatssDetector``) and ``["Rbeast"]``
  (``BeastDetector``). A contract test checks the format and that the tag
  lists the modules the test suite skips a detector on.
* ``make test`` and ``make docs`` run the test and documentation jobs of the
  CI (``pytest --tb=short -q``; ``sphinx -W -E -b html docs docs/_build/html``).
  ``PYTEST_ARGS`` selects tests, ``DOCS_BUILD_DIR`` changes the output
  directory, ``PYTHON`` the interpreter.

* Reproduction tests: tests that reproduce a published result on a dataset
  downloaded on demand (pinned by its SHA-256 digest) go to
  ``tests/reproduction/`` and are marked ``reproduction``. The marker is
  deselected by default (``pytest -m reproduction`` runs them); the protocol
  is in ``tests/reproduction/README.md``.

* ``GreedyGaussianDetector`` returns the change point indices, as its
  ``returns_dense=True`` tag and its documentation announce: it returned one
  segment label per time point (``0, ..., 0, 1, ..., 1, ...``) since 0.2.0,
  which code reading the output as change point indices misinterpreted.

* ``EAggloDetector`` returns a ``numpy`` integer array instead of a Python
  list, and the deprecated ``HMMDetector`` returns integer labels instead of
  floats.

* ``InformationGainDetector`` stops its search when every candidate position
  (the multiples of ``step``) is already a change point, and then returns
  fewer than ``k_max`` change points: when ``k_max`` exceeded the number of
  candidates (``ceil(n / step) - 1``), it returned ``-1`` and ``0`` among the
  change points, for instance ``[-1, 0, 5, 10, 15, 20, 25]`` for
  ``k_max=10, step=5`` on 30 points. The other change points are unchanged.
* Code that no module of the package imports is removed: an unfinished Python
  port of AutoPlait, older BOCD and TIRE scripts, the former IGTS
  implementation, a MATLAB-engine wrapper of ESPRESSO, PaTSS scripts that
  import a ``baselines`` package foreign to tsseg, and the MATLAB reference
  sources of ESPRESSO and iCID. The wheel no longer ships these modules nor
  the four data files of the MATLAB sources (two ``.mat`` files of the iCID
  demo, two ``.txt`` notes of ESPRESSO). The deprecated aliases
  ``HdpHsmmDetectorV1``, ``HdpHsmmLegacyDetector`` and
  ``LegacyHdpHsmmDetector`` of ``tsseg.algorithms.hdp_hsmm`` are removed:
  use ``HdpHsmmDetector``.
* ``AutoPlaitDetector`` returns the segmentation that the AutoPlait binary
  writes: each time point gets the number of its regime, so that segments of
  the same regime share a label, and change points are the first index of a
  segment. The wrapper read the change points one index early (the last index
  of the previous segment), took the labels of ``segment.labels`` (one row per
  regime, whose second column is the regime's path in the split tree) and
  applied them to the segments in time order, padding with the last one: on
  regimes A-B-A it returned ``1 | 0 | 0`` with a single change point. It now
  raises ``RuntimeError`` when the binary's segments do not cover each time
  point exactly once.

* ``AutoPlaitDetector`` is tagged unsupervised (``capability:unsupervised``
  ``True``, ``capability:semi_supervised`` ``False``), as the AutoPlait binary
  always ran: it chooses the number of segments itself and has no guided mode.
  It runs without ``n_cps`` and without ``y``; it raised ``ValueError`` when
  both were missing. ``n_cps`` is deprecated and ignored: passing it, ``0``
  included, emits a ``FutureWarning`` at ``fit`` and no longer launches the
  binary once with a number of segments that it rejected. The output does not
  change.

* ``AutoPlaitDetector`` hands the AutoPlait binary a series that it reads with
  the right length. The binary counts the time points of its input with a line
  buffer of 254 bytes, and the wrapper wrote 25 characters per value
  (``%.18e``): rows of 10 channels or more were counted twice or more, and the
  binary segmented a longer series padded with rows of zeros (62,442 time
  points instead of 31,392 on ActRecTut subject 1), whose last segment ended
  past the series. Values are now written as float32 with 9 significant
  digits, the precision at which the binary reads them, so that series of up
  to 15 channels always fit. A series whose rows still do not fit (the 76
  channels of JIGSAWS) raises ``ValueError`` before the binary runs, and a
  binary that reports another length than the series' raises
  ``RuntimeError``. Series of fewer than 10 channels were not affected.

* The AutoPlait runner (``AutoPlait._run_autoplait`` in
  ``tsseg.algorithms.autoplait.autoplait_c``) runs the binary once, with the
  command line of the bundled binary (``autoplait d list_file out_dir``), and
  raises ``RuntimeError`` if it fails. It no longer takes ``n_cps``, nor tries
  a command line with ``n_cps + 1`` or with the length of the series: the only
  binary that takes a second argument, the modified version of the FLOSS
  project page, reads it as a minimum number of regimes, not of segments. The
  documentation now points to the original C code of Matsubara et al., which
  ``c/autoplait/`` holds, instead of the FLOSS page. ``AutoPlaitDetector``
  returns the same output.

* The output contract tests check what the documentation states: change
  points are a 1-D integer ``numpy`` array, strictly increasing, within
  ``[1, n - 1]``; state labels are a 1-D integer ``numpy`` array of length
  ``n``. They accepted ``0``, duplicates and float labels.

* Guided mode on a series without any change point: ``ClaspDetector``
  (``n_change_points=0`` or ``n_segments=1``), ``ClapDetector``
  (``n_change_points=0``), ``EspressoDetector`` (``n_segments=1``),
  ``Time2StateDetector``, ``E2USDDetector`` and ``TiccDetector``
  (``n_states=1``) accept the value and return no change point (a single
  state); their parameter validation rejected it.

* ``AmocDetector`` computes the error of every split from cumulative sums, in
  linear time: it recomputed both means at every split (quadratic time).
  The detected change point is the same.

* ``fit`` and ``fit_predict`` reject a ``y`` given as a ``pandas.DataFrame``
  that does not have exactly one column, as the error message states: a
  frame with two columns (or none) was accepted.

* ``ChangeFinderDetector`` without ``n_cps``: the data-driven threshold is
  ``mean + threshold_factor * std`` of the scores (``threshold_factor``, 2.0
  by default as before), the minimum distance between change points can be a
  fraction of the length (``min_distance_fraction``), and the ``ensembling``
  strategy merges the detections of the channels that lie within the
  tolerance (by default the minimum distance) and can require a share of the
  channels to agree (``consensus``, 0 by default); it returned every distinct
  peak of every channel. The ``l2`` strategy and the behaviour with ``n_cps``
  are unchanged by default.

* ``FLUSSDetector`` accepts ``n_segments=None`` (number of segments unknown):
  the regime extraction keeps the minima of the corrected arc curve below
  ``threshold`` (0.5), and the ``ensembling`` strategy merges the detections
  of the channels within ``tolerance`` (``consensus``, the share of the
  channels that must agree, is 0 by default). A known ``n_segments`` behaves
  as before.

* New helper ``tsseg.algorithms.utils.consensus_change_points``: change points
  detected by enough channels, without a target count.

* Documentation matches the code: the tag list of the detectors page
  (``returns_dense`` is ``True`` for change point indices; guided mode is the
  ``capability:semi_supervised`` tag), the getting-started guide (examples that
  run without optional dependencies, guided mode through constructor
  parameters, real test commands), the datasets page (one bundled dataset, no
  test fixture to import), the extras table, change point examples that named
  their output ``labels``, and the citation (Springer proceedings).

* Docstrings: ``PeltDetector``, ``BottomUpDetector``, ``WindowDetector``,
  ``DynpDetector`` and ``KCPDDetector`` document their parameters;
  ``BinSegDetector`` states its actual ``penalty`` default (10; pass
  ``penalty=None`` to stop on ``epsilon``); the ``HidalgoDetector`` example
  runs with tsseg instead of aeon, and its ``K_states`` default is 1; the
  ``ICIDDetector`` ``psi_list`` default is ``None``. The ``compute`` methods of
  the state detection metrics and the ``validate`` methods of the parameter
  constraints are documented. A test checks that each detector documents
  exactly its constructor parameters, with their actual defaults.

* ``EAggloDetector``: ``penalty="mean_diff_penalty"`` no longer raises (the
  penalty was compiled with numba, which rejects the list of segment boundaries
  it receives). The documentation now describes the goodness-of-fit statistic,
  the V-statistics and the merge of the first and last clusters inherited from
  ``ecp``, and the over-segmentation obtained from one cluster per time point.

* ``ClaspDetector`` runs when ``NUMBA_NUM_THREADS`` is below the number of cores
  (as set by users or by SLURM jobs): ``n_jobs=-1`` meant ``os.cpu_count()``
  threads, more than numba accepts, which raised ``ValueError: The number of
  threads must be between 1 and ...``. ``n_jobs=-1`` now means all the threads
  numba may use, and a larger ``n_jobs`` is capped at that number.

* ``TireDetector``: the minimum distance between two change points is
  ``peak_distance_fraction`` times the series length, as documented; the default is 0 (no
  constraint). The prominence threshold of
  the unsupervised mode is the ``prominence_threshold`` parameter,
  0.5 by default.

* **Removed** ``VQTSSDetector`` and ``VSAXDetector``, two methods developed for
  tsseg without a publication. On a 54-series sample of TSB-SEG, VSAX scored
  below a random segmentation (bidirectional covering 0.351 against 0.377, ARI
  0.163 against 0.208), and VQTSS returned one vector-quantisation code per
  time step rather than a segmentation (ARI 0.001), with more than 10 minutes
  per series beyond 2,500 points. Both labelled local patterns rather than
  regimes. Code that imports them must drop them; they remain in the history
  of the repository and in tsseg 0.2.0.

* ``ProphetDetector`` now places ``n_candidates`` evenly spaced candidates, fits
  Prophet's rate changes ``delta`` and returns the ``n_changepoints`` largest
  ``|delta|`` relative to their median, or, when ``n_changepoints`` is ``None``
  (the new default), those above ``delta_threshold`` (a relative form of
  TCPDBench's rule), at least ``tolerance`` apart. New parameters
  ``n_candidates``, ``changepoint_prior_scale`` (500 by default, calibrated on
  TSB-SEG), ``delta_threshold``, ``seasonality`` (off by default) and
  ``max_points`` (longer series are reduced to block means before the fit);
  ``n_changepoints`` now accepts 0; ``multivariate_strategy`` defaults to
  ``"l2"``, and ``"ensembling"`` averages the per-channel scores instead of
  voting.

* The vendored ruptures solvers have a numba backend for the ``l1``, ``l2``,
  ``rbf`` and ``cosine`` costs, with the same segmentation as the Python path:
  ``backend="auto"`` (the default) picks it when numba is installed, ``"numba"``
  and ``"python"`` force either path, in ``PeltDetector``, ``DynpDetector``,
  ``BinSegDetector``, ``BottomUpDetector``, ``WindowDetector`` and
  ``KCPDDetector``. The kernel costs no longer build the n x n Gram matrix (for
  PELT and DynP, a transcription of the C solvers of ruptures' ``KernelCPD``);
  DynP no longer memoises O((n / jump)^2) sub-problems; BinSeg finds the best
  split of a segment in one sweep from each end.
* ``SNLDSDetector`` is now shipped with the package. It was listed in
  ``tsseg.algorithms`` without its module, so ``from tsseg.algorithms import *``
  failed on the 0.2.0 wheel. The authors' TensorFlow code (Google Research,
  Apache-2.0) is vendored in ``tsseg/algorithms/snlds/vendor``; install the
  dependencies with ``pip install tsseg[snlds]``. Its constructor is now
  keyword-only, and its tags declare it semi-supervised (``n_states`` is the
  number of regimes) rather than unsupervised. The wheel now also ships the
  ``LICENSE`` files of the vendored code.

* ``CostRbf``: the median heuristic no longer counts the distance of a sample to
  itself, whose rounding residues shifted gamma by about 1e-4, and no longer
  builds the Gram matrix.
* Ties: every vendored solver treats values closer than ``1e-9`` of the costs
  involved (at least of the signal's cost per sample, ``tie_unit``) as equal, and
  decides between them by position as upstream decides between exactly equal
  values: earliest change point (PELT, DynP), last change point of a segment and
  first segment (BinSeg), leftmost merge (BottomUp), no peak on a plateau and
  latest tied peak first (Window); a value tied with ``pen`` or ``epsilon``
  stops, and PELT keeps a start tied with its pruning bound (pruning less never
  loses the optimum).
* ``PatssDetector`` is deprecated (``FutureWarning``) and will be removed in a
  future release: it needs ``npbad``, which is not on PyPI, and over-segments
  (17 times more change points than the ground truth on a 54-series sample of
  TSB-SEG). The authors maintain PaTSEmb, which needs Java.

* ``WindowDetector`` searches its peaks as ruptures does
  (``argrelmax(mode="wrap")``): the vendored peak search cut the neighbourhood
  of a score at the ends of the signal, and could add a change point next to the
  first or last window.
* The vendored ruptures costs convert the signal to float64 in ``fit``: float32
  signals were costed in float32, and ``CostCosine`` failed on integers.
* ``BeastDetector`` uses the ``Rbeast`` package from PyPI, version 0.1.25 or
  later (new extra ``tsseg[beast]``), instead of the copy of Rbeast 0.1.15
  vendored in ``c/Rbeast``, which never returned on univariate series longer
  than about 4,600 points. Change points were one sample late (Rbeast counts
  time from 1); they are now the index of the first sample of the new segment.
  ``mcmc_seed`` defaults to 1: with 0, Rbeast draws a random seed and the
  output changed from one call to the next.

* Third-party licenses: ``bocd/``, ``ticc/``, ``time2state/`` and the
  aeon-derived ``eagglo/``, ``hidalgo/``, ``hmm/`` and ``igts/`` now ship a
  copy of their upstream license, in the source tree and the wheel. The code
  adapted in ``bocd/`` is under the MIT license, not Apache-2.0 as
  documented. The README lists ``BeastDetector``'s dependency Rbeast, under
  the license of its R package (GPL >= 2), and no longer claims that every
  vendored directory contains a license file.

* ``pen_scale="bic"`` in ``PeltDetector``, ``BinSegDetector``,
  ``BottomUpDetector`` and ``WindowDetector``: the penalty is a coefficient on
  ``log(n) * u``, u the cost of one sample of the whole signal (d for ``l2`` on
  unit-variance channels, where it is the BIC penalty; the mean absolute
  deviation for ``l1``; 1 for the kernel costs), so that one value applies
  across lengths, dimensions and costs.
* New ``GaussianHMMDetector``: a hidden Markov model with diagonal Gaussian
  emissions, fitted by Baum-Welch (EM) from k-means initialisations and decoded
  by Viterbi. The number of states is given (``n_states``) or chosen by BIC.
  ``HMMDetector``, which only decoded with user-supplied parameters, is
  deprecated (``FutureWarning``).

* ``BinSegDetector``, ``BottomUpDetector`` and ``WindowDetector`` accept
  ``n_cps=0`` again (no change point), as ``DynpDetector`` and ``KCPDDetector``
  do: the parameter schema required ``n_cps >= 1``, which failed the
  semi-supervised runs on series without change points.
* ``SNLDSDetector`` follows the training schedules of the paper: the
  temperature of the discrete transitions and the weight of the cross-entropy
  regulariser start at 1,000 (``temperature_init``, ``xent_init``) and decay as
  in the Reacher experiment, scaled to ``n_train_steps``. Before, both stayed
  inactive (temperature 1, weight 1e-10). Training draws ``batch_size`` windows
  of ``train_window`` points per step (new parameter, default 256; ``None`` for
  the whole series), so a step no longer runs over the whole series, and the
  defaults are ``batch_size=8`` and ``n_train_steps=2000``. Prediction is
  compiled, about 100 times faster on MoCap. The detector declares a parameter
  schema and the ``non_deterministic`` tag, and no longer reseeds NumPy's global
  generator. SNLDS is documented as experimental: on a sample of TSB-SEG it
  scores below a random segmentation.

* ``KCPDDetector`` is deprecated (``FutureWarning``): it is ``PeltDetector`` or
  ``DynpDetector`` with a kernel cost (``model="rbf"``, ``"cosine"``, or ``"l2"``
  for the linear kernel).
* ``PatssDetector`` runs with scikit-learn 1.8, which removed
  ``LogisticRegression(multi_class=...)`` and raised a ``TypeError``: the
  one-vs-rest elastic-net logistic regression of PaTSS is now
  ``OneVsRestClassifier(LogisticRegression(...))``, the same model (predicted
  probabilities equal to 1e-15 under scikit-learn 1.6).

* ``F1Score`` gives 1.0 when neither the ground truth nor the prediction has a
  change point and the series length is read from ``y_true``:
  ``F1Score().compute([n], [])`` gave 0.0, because the empty prediction was
  rejected before the boundaries were set aside (``compute([n], [n])`` and
  ``compute([n], [], n_timepoints=n)`` already gave 1.0). Scores computed from
  labels (``convert_labels_to_segments=True``) are unchanged. The warning about
  a margin rounded to 0 samples is only emitted when change points are
  matched.

* ``F1Score``, ``Covering``, ``BidirectionalCovering`` and ``GaussianF1Score``
  accept the series length in ``compute(y_true, y_pred, n_timepoints=n)``;
  the change points may then include or omit ``0`` and ``n``, so the raw
  output of a detector can be scored directly. Without ``n_timepoints`` (and
  without ``convert_labels_to_segments=True``), the length is still read from
  the change points, scores are unchanged, and a ``UserWarning`` now says so:
  a ``y_true`` that does not end with the series length silently lost its last
  change point (``F1Score(margin=5).compute([300, 700], [300])`` gives 1.0;
  with ``n_timepoints=1000``, 0.667).

* Detector pages give the time complexity of 18 more detectors, as
  implemented with their default settings. Two values were wrong: PELT is
  :math:`O(C\,n)` expected and :math:`O(C\,n^{2})` at worst, with no factor in
  the number of change points (the page said :math:`O(C\,K\,n)`), and
  ChangeFinder solves one :math:`k \times k` system per sample, so
  :math:`O(d\,n + n\,k^{3})` (the page said :math:`O(n\,k^{2})`).

* The tGLAD page gives the default of ``epochs`` (100; it said 2000) and
  explains why the paper reports :math:`O(n)` where the page states
  :math:`O(d^{3}\,n)` per epoch. The ``TGLADDetector`` docstring links to the
  authors' repository (``Harshs27/tGLAD``).

0.2.0 (2026-09-26)
------------------

* Replaced the NumPy ``HdpHsmmDetector`` with a corrected sampler equivalent to
  ``pyhsmm``'s ``WeakLimitHDPHSMM`` (auxiliary self-transitions of Johnson & Willsky,
  right-censored last segment, chain started from the prior, numba forward messages)
  and calibrated defaults.
* ``numba`` is optional again, as ``tsseg[accelerators]`` announces: ``ClapDetector``,
  ``ClaspDetector`` and ``EAggloDetector`` are imported lazily (they need numba), and
  ``HdpHsmmDetector`` falls back to interpreted code without it.
* ``EAggloDetector`` and ``GreedyGaussianDetector`` declare ``returns_dense=True``,
  consistent with their change point output.
* ``KCPDDetector``: ``decimation`` parameter, ``pen_scale="bic"`` (penalty as a
  coefficient on ``log(n) * d``), and ``n_cps=0`` accepted.

0.1.1 (2026-04-20)
------------------

* Added ``BeastDetector`` and ``ChangeFinderDetector``.
* ``InformationGainDetector`` (IGTS) extended to univariate series, with the
  optimisation of the paper.
* Parameter schemas for all detectors; GF1 score exposed.
* Fixes to GLAD, Hidalgo and VSAX; continuous integration (lint, tests, docs).

0.1.0 (2026-02-10)
------------------

Initial public release.

* Vendored a lightweight subset of ``ruptures`` v1.1.8 — BinSeg, BottomUp,
  DynP, KCPD, PELT and Window detectors no longer need an external dependency.
* Added ``BottomUpDetector`` and ``DynpDetector`` wrappers with aeon-compatible
  interfaces.
* Replaced the surrogate PyTorch ``TSCP2Detector`` with a faithful
  TensorFlow + TCN implementation matching the original baseline.
* Replaced the ``pyhsmm``-backed ``HdpHsmmDetector`` with a native
  NumPy/SciPy sticky HDP implementation. ``pyhsmm`` and ``pybasicbayes`` are
  no longer installed by default.
* Improved ``ClaspDetector`` axis handling.
* Added Sphinx documentation with GitHub Pages deployment.
* Documented test suite layout and fixtures.
