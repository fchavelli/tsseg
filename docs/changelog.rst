:orphan:

.. _changelog:

Changelog
=========

This changelog highlights notable updates. For full commit history, refer to
`GitHub <https://github.com/fchavelli/tsseg/commits/main>`_.

Unreleased
----------

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

* ``F1Score``, ``Covering``, ``BidirectionalCovering`` and ``GaussianF1Score``
  accept the series length in ``compute(y_true, y_pred, n_timepoints=n)``;
  the change points may then include or omit ``0`` and ``n``, so the raw
  output of a detector can be scored directly. Without ``n_timepoints`` (and
  without ``convert_labels_to_segments=True``), the length is still read from
  the change points, scores are unchanged, and a ``UserWarning`` now says so:
  a ``y_true`` that does not end with the series length silently lost its last
  change point (``F1Score(margin=5).compute([300, 700], [300])`` gives 1.0;
  with ``n_timepoints=1000``, 0.667).

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
