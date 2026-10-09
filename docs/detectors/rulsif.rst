RuLSIF
======

Change-point detection by relative density-ratio estimation.

Description
-----------

The series is cut into overlapping subsequences of length :math:`k`
(``subsequence_length``); a multivariate subsequence concatenates all
channels. For every candidate split, the :math:`n` subsequences before it
(distribution :math:`P`) are compared with the :math:`n` subsequences after it
(distribution :math:`P'`, ``n_subsequences``) by the symmetrised
:math:`\alpha`-relative Pearson divergence

.. math::

   \mathrm{PE}_\alpha(P \| P') + \mathrm{PE}_\alpha(P' \| P), \qquad
   \mathrm{PE}_\alpha(P \| P') = \frac{1}{2} \int p'_\alpha(Y)
   \left(r_\alpha(Y) - 1\right)^2 \mathrm{d}Y,

with :math:`p'_\alpha = \alpha p + (1 - \alpha) p'` and the relative density
ratio :math:`r_\alpha = p / p'_\alpha`. Each divergence is estimated without
estimating densities: RuLSIF fits :math:`r_\alpha` by regularised least
squares with a Gaussian kernel model centred on the samples of :math:`P`,
which has a closed-form solution. As in the authors' code, the subsequences
of the two sets are first standardised coordinate by coordinate, so that
channels on different scales weigh the same; the kernel width (a multiple of
the median distance between subsequences) and the regularisation parameter are
chosen by 5-fold cross-validation for every estimate. The relative ratio is bounded by
:math:`1 / \alpha`, which makes the estimate more stable than that of the plain
ratio (:math:`\alpha = 0`, uLSIF).

Change points are the peaks of this score at least ``min_distance`` points
apart: the ``n_cps`` highest ones in guided mode, those above ``threshold``
otherwise. The paper only evaluates the score over all thresholds (ROC
curves); the default ``threshold`` (2.25, for :math:`\alpha = 0.1`) is
calibrated by tsseg: it maximises the mean F1 score (margin 1 %) over the
paper's four synthetic datasets. The score saturates near its bound
:math:`(1 - \alpha) / \alpha` on smooth or drifting series (motion capture,
random walk), where no threshold separates the changes: use ``n_cps`` there.
The score is kept in the ``scores_`` attribute after ``predict``.

| **Type:** change point detection
| **Supervision:** unsupervised (``threshold``) or semi-supervised (``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(T \cdot F S L \cdot n^3)` for a series of length
  :math:`T`, :math:`F` folds, :math:`S` kernel widths and :math:`L`
  regularisation parameters; about 40 s for 5,000 points with the defaults
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
   * - ``subsequence_length``
     - int
     - ``10``
     - Subsequence length :math:`k` (paper value).
   * - ``n_subsequences``
     - int
     - ``50``
     - Subsequences per compared set, :math:`n` (paper value). The series
       needs at least :math:`2n + k - 1` points.
   * - ``alpha``
     - float
     - ``0.1``
     - Mixture weight of the relative density ratio, in :math:`[0, 1)`
       (paper value).
   * - ``sigma_factors``
     - tuple of float
     - ``(0.6, 0.8, 1.0, 1.2, 1.4)``
     - Candidate kernel widths, as multiples of
       :math:`\sqrt{\operatorname{median}(d^2) / 2}` over the pairwise
       distances :math:`d` between subsequences: the paper's factors, the
       authors' code's base width (the paper's median distance over
       :math:`\sqrt 2`).
   * - ``lambdas``
     - tuple of float
     - ``(1e-3, 1e-2, 1e-1, 1, 10)``
     - Candidate regularisation parameters (paper values).
   * - ``n_folds``
     - int
     - ``5``
     - Cross-validation folds (paper value).
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points; ``None`` thresholds the peaks.
   * - ``threshold``
     - float
     - ``2.25``
     - Minimum peak score without ``n_cps``, calibrated for
       :math:`\alpha = 0.1` (tsseg). The score lies in
       :math:`[0, (1 - \alpha) / \alpha]`: scale the threshold with this
       bound for another :math:`\alpha`.
   * - ``min_distance``
     - int
     - ``20``
     - Minimum distance between change points; the higher of two closer
       peaks is kept (the paper drops alarms closer than 20 points to the
       previous one).
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import RuLSIFDetector

   # peaks above the threshold
   cps = RuLSIFDetector().fit_predict(X)

   # known number of change points
   detector = RuLSIFDetector(n_cps=3)
   cps = detector.fit_predict(X)
   score = detector.scores_

**Implementation:** NumPy port of the authors' MATLAB code
(`anewgithubname/change_detection <https://github.com/anewgithubname/change_detection>`_,
MIT licence), checked against a line-by-line transcription of it; it
reproduces the code's demo on well-log data (reproduction test).

**Reference:** Liu, Yamada, Collier & Sugiyama (2013), *Change-point detection
in time-series data by relative density-ratio estimation*, Neural Networks 43;
Yamada, Suzuki, Kanamori, Hachiya & Sugiyama (2013), *Relative density-ratio
estimation for robust distribution comparison*, Neural Computation 25(5).

API reference
-------------

.. automodule:: tsseg.algorithms.rulsif.detector
   :members:
   :show-inheritance:
   :undoc-members:
