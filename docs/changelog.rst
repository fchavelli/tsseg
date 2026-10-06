:orphan:

.. _changelog:

Changelog
=========

This changelog highlights notable updates. For full commit history, refer to
`GitHub <https://github.com/fchavelli/tsseg/commits/main>`_.

Unreleased
----------

* ``TireDetector``: the minimum distance between two change points is
  ``peak_distance_fraction`` times the series length, as documented; the default is 0 (no
  constraint). The prominence threshold of
  the unsupervised mode is the ``prominence_threshold`` parameter,
  0.5 by default.

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
* ``WindowDetector`` searches its peaks as ruptures does
  (``argrelmax(mode="wrap")``): the vendored peak search cut the neighbourhood
  of a score at the ends of the signal, and could add a change point next to the
  first or last window.
* The vendored ruptures costs convert the signal to float64 in ``fit``: float32
  signals were costed in float32, and ``CostCosine`` failed on integers.
* ``pen_scale="bic"`` in ``PeltDetector``, ``BinSegDetector``,
  ``BottomUpDetector`` and ``WindowDetector``: the penalty is a coefficient on
  ``log(n) * u``, u the cost of one sample of the whole signal (d for ``l2`` on
  unit-variance channels, where it is the BIC penalty; the mean absolute
  deviation for ``l1``; 1 for the kernel costs), so that one value applies
  across lengths, dimensions and costs.
* ``BinSegDetector``, ``BottomUpDetector`` and ``WindowDetector`` accept
  ``n_cps=0`` again (no change point), as ``DynpDetector`` and ``KCPDDetector``
  do: the parameter schema required ``n_cps >= 1``, which failed the
  semi-supervised runs on series without change points.
* ``KCPDDetector`` is deprecated (``FutureWarning``): it is ``PeltDetector`` or
  ``DynpDetector`` with a kernel cost (``model="rbf"``, ``"cosine"``, or ``"l2"``
  for the linear kernel).

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
