SST
===

Change-point detection by singular spectrum transformation.

Description
-----------

The singular spectrum transformation (SST) compares the local dynamics of the
series before and after each time point. The lagged vectors
:math:`s(j) = (x_{j - N + 1}, \dots, x_j)` of length :math:`N`
(``window_size``) form the past Hankel matrix
:math:`H_1(t) = [s(t - N), \dots, s(t - 1)]` and the future one
:math:`H_2(t) = H_1(t + \gamma)`, :math:`\gamma` points later (``lag``). The
:math:`r` leading left singular vectors :math:`u_1(t), \dots, u_r(t)` of
:math:`H_1(t)` (``rank``) span the dominant subspace of the past, a linear
model of its dynamics; :math:`\mu(t)`, the leading one of :math:`H_2(t)`, is
the main direction of the future. The score

.. math::

   z(t) = 1 - \sum_{i=1}^{r} \left(u_i(t)^\top \mu(t)\right)^2 \in [0, 1]

is the squared distance between :math:`\mu(t)` and the past subspace (Idé &
Tsuda, 2007). It rises when the shape of the signal changes (frequency or form
of an oscillation) and, after the preprocessing below, at level shifts;
changes of the variance or of the autocorrelation of noise are found late or
not at all. Every channel is first brought to unit variance and mean 3, as in
the papers: the SVD is not invariant to translation, and the positive offset
makes the leading singular value non-degenerate. A multivariate series puts
the Hankel matrices of its channels side by side (one shared subspace, as
multivariate SSA does with Page matrices).

By default the singular vectors come from the randomised SVD of Weber & Lenz
(2025), which multiplies the Hankel matrices by FFT instead of forming them:
:math:`O(q (k + p) N \log N)` per time point instead of :math:`O(N^3)` for a
full SVD (``method="exact"``), for a score error that the paper bounds and
measures (0.036 on average on its synthetic series, without the preprocessing
above, which makes the error much smaller).

The papers give the score only. Change points are the ``n_cps`` highest peaks
of the score at least ``min_distance`` points apart; there is no
unsupervised mode, and ``n_cps`` is required. A peak at :math:`t` is reported
as a change at :math:`t`, the first point after the past matrix; the score
peaks within a few points of a level shift, up to about :math:`N` points after
a frequency change, and several :math:`N` after a change of the variance of
white noise. The score is 0 where the two
matrices do not fit (the first :math:`2N - 1` points and the last
:math:`\gamma - 1`) and is kept in the ``scores_`` attribute after
``predict``.

| **Type:** change point detection
| **Supervision:** semi-supervised (``n_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(T \cdot q (k + p)\, d N \log N)` for :math:`T`
  points and :math:`d` channels (``"fft_rsvd"``), :math:`O(T d N^3)`
  (``"exact"``); about 5 s for 20,000 points with the defaults
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
   * - ``window_size``
     - int
     - ``50``
     - Window :math:`N`: length of the lagged vectors and number of columns
       of each Hankel matrix. A score needs :math:`2N - 1 + \gamma` points.
       The papers fix no value; 50 is that of Idé & Tsuda's experiments.
   * - ``lag``
     - int / None
     - ``None``
     - Lag :math:`\gamma` between the past and future matrices; ``None``
       means ``window_size // 2`` (Idé & Tsuda).
   * - ``rank``
     - int
     - ``5``
     - Dimension :math:`r < N` of the past subspace (Weber & Lenz, for
       windows of 100 to 5,000 points; Idé & Tsuda recommend 3 or 4).
   * - ``oversampling``
     - int
     - ``10``
     - Oversampling :math:`p` of the randomised SVD (Weber & Lenz).
   * - ``n_iter``
     - int
     - ``3``
     - Subspace iterations :math:`q` of the randomised SVD (Weber & Lenz).
   * - ``method``
     - str
     - ``"fft_rsvd"``
     - ``"fft_rsvd"`` (randomised SVD, FFT Hankel products) or ``"exact"``
       (full SVD).
   * - ``n_cps``
     - int / None
     - ``None``
     - Number of change points (required).
   * - ``min_distance``
     - int / None
     - ``None``
     - Minimum distance between change points; ``None`` means
       ``window_size``.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed of the randomised SVD.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import SSTDetector

   detector = SSTDetector(n_cps=3, random_state=0)
   cps = detector.fit_predict(X)
   score = detector.scores_

**Implementation:** written for tsseg from the papers, without third-party
code. On synthetic signals of the four kinds of Weber & Lenz, scored raw as
in the paper, the mean absolute error of the randomised SVD against the full
SVD stays below the paper's figure (reproduction test).

**Reference:** Idé & Tsuda (2007), *Change-point detection using Krylov
subspace learning*, SDM; Idé & Inoue (2005), *Knowledge discovery from
heterogeneous dynamic systems using change-point correlations*, SDM; Weber &
Lenz (2025), *Accelerating singular spectrum transformation for scalable
change point detection*, IEEE Access 13.

API reference
-------------

.. automodule:: tsseg.algorithms.sst.detector
   :members:
   :show-inheritance:
   :undoc-members:
