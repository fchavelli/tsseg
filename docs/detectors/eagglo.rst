E-Agglo
=======

E-Agglo – energy-based agglomerative change point detection.

Description
-----------

E-Agglo is a non-parametric, hierarchical agglomerative algorithm for detecting
multiple change points in multivariate time series.  Adjacent segments are
merged greedily to maximise a goodness-of-fit statistic, the sum of the energy
divergences between adjacent segments, and the segmentation with the largest
statistic along the merging sequence is returned.  Unlike classical
agglomerative clustering, this procedure preserves the temporal ordering.

A divergence parameter :math:`\alpha\in(0,2]` controls the distance exponent
(:math:`\alpha = 2` only compares means).  From one cluster per point
(``member=None``, the default) the statistic is usually maximised by a very
fine segmentation: start from blocks of a few samples (``member``) and/or set a
penalty.  The method assumes independent observations and can over-segment
serially dependent series whatever the initial partition.

The implementation follows ``ecp`` rather than reproducing every convention of
the paper: within-cluster distances include self-pairs (V-statistics), and the
first and last clusters are neighbours and may merge. These choices affect the
objective and cannot be removed by accelerating the code.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(n^{2})`
| **Requires:** numba

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``member``
     - array-like / None
     - ``None``
     - Initial cluster of each point (sorted labels, contiguous blocks). ``None`` = one cluster per point.
   * - ``alpha``
     - float
     - ``1.0``
     - Divergence exponent in :math:`(0, 2]`.
   * - ``penalty``
     - str / callable / None
     - ``None``
     - Penalty function (``"len_penalty"``, ``"mean_diff_penalty"`` or callable).

Usage
-----

.. code-block:: python

   import numpy as np
   from tsseg.algorithms import EAggloDetector

   member = np.arange(len(X)) // 10  # initial blocks of 10 samples
   detector = EAggloDetector(alpha=1.0, member=member)
   change_points = detector.fit_predict(X)

**Implementation:** Adapted from aeon.  BSD 3-Clause.

**Reference:** Matteson & James (2014), *A Nonparametric Approach for Multiple
Change Point Analysis of Multivariate Data*, JASA.

Runtime and memory
------------------

For :math:`n` observations of dimension :math:`d` and :math:`m` initial blocks,
distance initialisation takes :math:`O(n^2d)`, even with blocks: each block pair
still compares all its observations. The merge search takes :math:`O(m^2)`.
Blocks restrict the candidate boundaries; they do not downsample the data.

The distance matrix has shape ``(2*m, 2*m)`` and the boundary history has shape
``(m, m+1)``. Together their float64 storage is approximately
:math:`40m^2` bytes, excluding temporary arrays and other fitted attributes:
about 4 GB for 10,000 singleton blocks and 400 GB for 100,000. The exact method
is consequently unsuitable for long series with the singleton default.

The distance initialisation and complete candidate-merge scans run in Numba.
Earlier versions called a small compiled kernel from Python for every pair and
candidate, and also scanned the full membership vector once per singleton.
Moving these loops into Numba removes substantial overhead without changing the
statistic, merge tie rule, penalties or segmentation. On one local run with
4,000 four-dimensional Gaussian observations, after initial compilation, the
fit and prediction took 1.67 s instead of 18.06 s. This is a timing example,
not a complexity improvement; the first fit also pays compilation costs.

Why fine segmentations can win
------------------------------

Under independent sampling from one distribution, let
:math:`\delta=\mathbb{E}\|X-X'\|^\alpha`. The V-statistic estimate of the
energy divergence between groups of sizes :math:`a,b` has expectation
:math:`\delta(1/a+1/b)`. Its scaled version
:math:`ab/(a+b)` times the divergence therefore has positive expectation
:math:`\delta`, even when there is no change. Adding boundaries can increase
the unpenalised objective just by adding this sampling contribution.

With 1,000 independent four-dimensional standard Gaussian observations and
``np.random.default_rng(0)``, the singleton default returns 959 change points.
On the existing three-segment mean-and-variance example (seed 0, 300 points),
it returns 192 instead of two. Initial blocks of ten samples recover the two
boundaries exactly in the latter example.

``len_penalty`` subtracts one per boundary, with no adjustable coefficient.
The objective scales as :math:`|c|^\alpha` when all observations are multiplied
by :math:`c`; this fixed penalty does not. A callable can supply a coefficient,
but its scale needs calibration. The penalty selects a step along the same
greedy merge sequence; it cannot recover a boundary already removed before
that step. Neither named penalty is an automatic statistical calibration for
serial dependence or arbitrary measurement units.

There is also a scale convention inherited from ``ecp``: initial block
divergences omit the sample-size factor :math:`ab/(a+b)`, while subsequent
updates use the recurrence for scaled divergences. For equal initial block
size :math:`b`, the circular statistic is :math:`4/b` times the sum of scaled
V-divergences. Unequal block sizes do not admit a common scale factor. Even a
uniform grid can end with a shorter last block. This convention, and the
circular adjacency, differ from a literal implementation of equations (6)
and (13) in the `original paper <https://arxiv.org/abs/1306.4933>`_.

Diagnosing suitability for a series
-----------------------------------

E-Agglo compares the distributions of observation vectors. It has no explicit
model of a motion cycle, waveform or within-segment autocorrelation. Changes
of temporal dynamics with the same marginal distribution are therefore not
directly identified by the energy divergence on raw observations. The
consistency guarantee in the original paper concerns E-Divisive, not E-Agglo.

For example, on the bundled MoCap trial ``86_01`` (4,579 points, four channels,
six annotated changes), without preprocessing:

.. list-table::
   :header-rows: 1
   :widths: 45 15 20 20

   * - Input and initial partition
     - Detected changes
     - F1 (1% margin)
     - BiCovering
   * - Raw observations, singletons
     - 39
     - 0.133
     - 0.212
   * - Raw observations, blocks of 10
     - 40
     - 0.130
     - 0.246
   * - Raw observations, blocks of 50
     - 39
     - 0.133
     - 0.223
   * - Raw observations, blocks of 50, ``len_penalty``
     - 39
     - 0.133
     - 0.223
   * - Shuffled within reference segments, blocks of 50
     - 6
     - 0.833
     - 0.951

The last row jointly permutes the observation vectors inside each annotated
segment, with seed 0. It preserves each segment's empirical multivariate
distribution while destroying its temporal structure. It uses the reference
labels and is a diagnostic only. This example suggests that temporal
structure and variation inside an activity, rather than just a poor initial
partition, drive the extra detections on this trial. It does not establish
performance on other datasets.

On all nine bundled MoCap trials, with per-channel z-normalisation and blocks
of 50, the mean BiCovering is 0.317 on the original observations and 0.785 after
the same within-reference-segment shuffling. The mean numbers of detected
changes are 60.7 and 5.3 respectively, against 7.2 annotated changes. Eight
trials improve under this diagnostic; one gets worse. The results indicate a
broader sensitivity on MoCap, without isolating serial dependence from other
variation inside an activity or establishing an automatic remedy.

Run these diagnostics from a source checkout:

.. code-block:: sh

   PYTHONPATH=. python examples/eagglo_diagnostics.py > /tmp/eagglo.jsonl

Add ``--all-mocap`` to reproduce the nine-trial comparison.

The script also measures runtime after initial compilation. It uses the
bundled raw MoCap trial, so its numbers are not a replay of a benchmark with
different preprocessing. The synthetic examples are diagnostics, not a
reproduction of the publication's experimental results. For the authors'
examples and implementation details, see the
`ecp paper <https://arxiv.org/abs/1309.3295>`_ and
`R implementation <https://github.com/cran/ecp/blob/master/R/e_agglomerative.R>`_.

A calibrated MoCap profile
--------------------------

For the bundled MoCap signals, a useful starting profile is channel-wise
z-normalisation, ``alpha=0.5``, initial blocks of 200 observations and
``penalty=None``. This keeps the existing detector and estimates the change
point count without receiving labels or a count:

.. code-block:: python

   import numpy as np
   from tsseg import load_mocap
   from tsseg.algorithms import EAggloDetector

   X, _ = load_mocap(trial=0, return_X_y=True)
   std = X.std(axis=0)
   X = (X - X.mean(axis=0)) / np.where(std == 0, 1, std)
   detector = EAggloDetector(
       alpha=0.5,
       member=np.arange(len(X)) // 200,
       penalty=None,
   )
   change_points = detector.fit_predict(X)

The profile was selected in an exploratory grid of 288 settings on the nine
original MoCap trials, without shuffling. The grid varied ``alpha`` in
``{0.5, 1, 2}``, block size in ``{10, 25, 50, 100, 200, 400}``, and 16 penalty
settings: eleven linear coefficients (including no penalty and
``len_penalty``), the paper's mean-length reward, and four distance-scaled
logarithmic penalties. The logarithmic choices are comparison heuristics,
not a derived information criterion. Both the grid and the final evaluation
use the benchmark's per-channel z-normalisation.

.. list-table:: Mean BiCovering on the nine original MoCap trials
   :header-rows: 1
   :widths: 20 25 25 30

   * - Alpha
     - Initial blocks
     - Penalty
     - Mean BiCovering
   * - 1.0
     - Singletons (default)
     - None
     - 0.272
   * - 1.0
     - 50 observations
     - None
     - 0.317
   * - 1.0
     - 200 observations
     - None
     - 0.649
   * - 0.5
     - 200 observations
     - None
     - 0.718

The chosen profile improves BiCovering on all nine trials. Its mean F1 at a
1% margin is 0.437, against 0.240 for the defaults. These are scores on the
same trials used to select the profile, not an independent test estimate.
In a leave-one-trial-out check, selecting among all 288 settings on eight
trials and scoring only the ninth gives mean BiCovering 0.688. Every fold
selects ``alpha=0.5`` and blocks of 200; eight select no penalty, and one
selects a linear coefficient of 0.25. Reference labels select hyperparameters
across trials; the held-out detector receives neither labels nor the true
number of changes.

Longer blocks substantially reduce over-segmentation on these signals.
Adding a fixed penalty can remove real transitions: it is not an automatic
improvement once the partition is coarser. The selected block size also
restricts every returned boundary to a multiple of 200 observations. It can
miss close changes and limits location precision, so this profile is intended
for MoCap, not as a universal replacement for the constructor defaults. The
F1 decreases on two trials despite a higher BiCovering; on ``86_09`` it falls
from 0.750 to zero because the coarse boundaries miss the 1% tolerance. The
paper motivates block initialisation and permits exponents below one; the
specific values here come from this calibration, not a recommendation made
by the authors.

To reproduce the grid, its cross-validation, and the final comparison:

.. code-block:: sh

   PYTHONPATH=. python examples/eagglo_mocap_calibration.py --sweep --verify

The script writes the full grid, ranking, held-out selections and verified
predictions under ``/tmp/eagglo-mocap-calibration`` (``--output`` changes this).
Exploration shares a SciPy pairwise-distance matrix across block sizes and
reuses merge histories across penalties. Verification recomputes the default
and proposed profiles with the public detector and its normal kernels.

API reference
-------------

.. automodule:: tsseg.algorithms.eagglo.detector
   :members:
   :show-inheritance:
   :undoc-members:
