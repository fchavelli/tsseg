Cachalot
========

Cachalot (CAtching CHAnge-points with LassO), also known as TV Lasso, detects
changes in the level of a piecewise-constant signal. It first selects
candidate change points along a Lasso path, then refines the segmentation
with dynamic programming restricted to those candidates. The number of
change points is chosen by the relative reduction in residual sum of squares.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate method; multivariate input is processed independently
  by channel
| **Complexity:** :math:`O(d (n K + K^3))` for *d* channels and *K* candidates per channel

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``max_candidates``
     - int
     - ``50``
     - Maximum number of Lasso-path candidate change points per channel,
       capped at ``n_timepoints - 1``.
   * - ``nu``
     - float
     - ``0.01``
     - Relative-improvement threshold. Select the first *k* such that
       ``J(k + 1) / J(k) >= 1 - nu``.
   * - ``axis``
     - int
     - ``0``
     - Axis of ``X`` that represents time.

The paper presents a univariate method. For multivariate input, each channel
gets its own Lasso path and reduced dynamic program; the detector returns the
union of their selected change points. This is a channel-wise extension, not a
shared-change-point model such as group LARS. Nearby detections from different
channels are not merged, and the union can contain up to ``d * max_candidates``
change points. Channel-wise scaling does not balance or combine evidence across
channels. A method for common ruptures is the
`group fused Lasso of Bleakley and Vert (2011) <https://arxiv.org/abs/1106.4199>`_.

On the nine four-channel MoCap
trials bundled with tsseg, the union raised macro F1 from 0.357 (mean of the
four single-channel scores) to 0.404 using a 1% matching margin; macro
precision and recall were 0.304 and 0.655. The mean per-trial best-channel
F1, an oracle reference, was 0.576. No predicted change point was exactly
shared by all four channels. This supports the union only as a high-recall
channel-wise extension, not as evidence of a shared multivariate model.

Implementation and validation
-----------------------------

The Lasso path exploits the inverse of the centered step-design Gram matrix:
its action on the active signs is linear interpolation between active split
positions. This computes the same unweighted TV-Lasso path in :math:`O(n K)`
time per channel, without a dense design or Gram matrix. Reduced DP still
searches all candidate-supported segment counts in :math:`O(K^3)` time.
Working memory is :math:`O(n + K^2)` per channel, plus stored results.

The implementation follows the unpenalized signal level in Eq. (4) of the
2007 paper. Step columns are centered but not standardized. Selection also
allows zero changes, as in its Table 1; if no ratio reaches the threshold,
the largest candidate-supported count is returned. Exact ties enter in
increasing index order and can be truncated by the candidate budget.

Against the original PR implementation, single-thread median times on 10,000
points fell from 0.428 s to 0.012 s for 50 candidates and from 3.76 s to
0.089 s for 200 candidates. All 36 single-channel MoCap predictions were
unchanged. These are local measurements, not performance guarantees.

Tests compare candidate paths against scikit-learn's dense Lasso path,
reduced-DP costs and reconstructed splits against exhaustive search, and the
selection rule against the published objective table. Regressions cover
constant decimal levels, changes of signal unit, tied entries, short inputs,
parameter validation, and both multivariate axis conventions.

A separate seeded study of 2,400 synthetic series uses the Section 5.1 model,
with explicitly different sampling and matching conventions. It does not
reproduce the published Monte Carlo tables: for five true changes and noise
multiplier 0.5, mean precision was 0.586 (paper: 0.80), with recall 0.999.
This discrepancy remains unresolved; the implementation's model-count rule
can oversegment. See the
`audit report <https://github.com/fchavelli/tsseg/blob/baseline/tv-lasso/tsseg/algorithms/cachalot/AUDIT.md>`_
and its reproducible script for protocols, results, and limitations.

The paper's toy example reports candidate locations and an objective table,
but omits the sampled noise vector and random seed. Its Monte Carlo tables
also omit seeds, so those exact random realizations cannot be regenerated.
The paper reports that its toy objective table selects four change points at
``(30, 50, 70, 90)`` with ``nu=0.05``.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import CachalotDetector

   detector = CachalotDetector(max_candidates=50, nu=0.01)
   change_points = detector.fit_predict(X)

**References:** Harchaoui, Z. and Lévy-Leduc, C., *Catching Change-points
with Lasso*, Advances in Neural Information Processing Systems 20, 2007;
Harchaoui, Z. and Lévy-Leduc, C., *Multiple Change-Point Estimation With a
Total Variation Penalty*, Journal of the American Statistical Association
105(492):1480-1493, 2010.

API reference
-------------

.. automodule:: tsseg.algorithms.cachalot.detector
   :members:
   :show-inheritance:
   :undoc-members:
