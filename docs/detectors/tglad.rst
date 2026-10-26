tGLAD
=====

tGLAD — temporal Graph-difference change point detection.

Description
-----------

tGLAD segments multivariate time series by tracking the evolution of
conditional independence (CI) graphs over time.  A sliding window converts the
series into overlapping intervals; the uGLAD sparse graph recovery model
(multitask mode) recovers a precision matrix (CI graph) for each interval
simultaneously.  Following the authors' code, the score of a window is the
absolute second difference, across windows, of the sum of its partial
correlations (for a univariate series, tsseg uses the precision of the single
channel).  In place of the allocation algorithm of the paper, tsseg returns the
start of each window whose score reaches ``threshold``, keeping change points
at least ``min_spacing`` apart.

Each batch of windows trains a new uGLAD model, and uGLAD scales each channel
of each window to :math:`[0, 1]` (min-max) first.  What to expect:

- **Training length.**  GLAD shrinks the off-diagonal entries of the precision
  matrix by a soft threshold that it learns.  After too few epochs, this
  threshold can set every partial correlation of a window to zero; when it
  does so in every window, all scores are 0 and no change point is found.
  With 100 epochs, this happened on a two-channel series with two clear mean
  shifts, and on 96 to 100 % of the windows of the first three MoCap trials
  (four channels, default ``window_size``).  ``epochs`` is therefore 2000 by
  default, as in the authors' code; training takes about 20 times longer
  than with 100.
- **Several change points per transition.**  The windows around a transition
  can each reach ``threshold`` and give a change point at their start.  The
  authors' code instead marks the ``window_size`` points on each side of each
  such window as a transition, so that the marked intervals merge.
- **Univariate series.**  Without partial correlations, the summary of a
  window is the precision of its single channel, a fallback that is not in
  the authors' code.  Because of the min-max scaling, it does not follow the
  variance of the signal, only the shape of the distribution of the window's
  values; after 2000 epochs it varies from one window to the next, within a
  stationary segment too, by far more than the default ``threshold``, so
  that most windows give a change point.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** multivariate; univariate accepted (fallback summary, see above)
| **Complexity:** :math:`O(d^{3}\,n)` per training epoch
| **Requires:** PyTorch, networkx

.. note::

   The tGLAD paper (Imani & Shrivastava, AALTD 2023, Sec. 3.3) reports
   :math:`O(n)` when :math:`d^{3} \ll n`. This counts :math:`d` and the number
   of epochs as constants and assumes the batches run in parallel; the
   sequential cost it derives, :math:`O(d^{3}\,n)` per epoch, is the one above.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 10 10 58

   * - Name
     - Type
     - Default
     - Description
   * - ``window_size``
     - int
     - ``512``
     - Time steps per uGLAD window.
   * - ``stride``
     - int
     - ``128``
     - Step between successive windows.
   * - ``batch_size``
     - int
     - ``8``
     - Windows processed together by the multitask solver.
   * - ``threshold``
     - float
     - ``0.5``
     - Min window score to trigger a change point.
   * - ``min_spacing``
     - int / None
     - ``None``
     - Min distance between emitted CPs (default: ``stride``).
   * - ``epochs``
     - int
     - ``2000``
     - uGLAD training epochs per batch (2000 in the authors' code).
   * - ``learning_rate``
     - float
     - ``0.001``
     - Optimiser learning rate.
   * - ``glad_iterations``
     - int
     - ``5``
     - Unrolled GLAD iterations.
   * - ``eval_offset``
     - float
     - ``0.1``
     - Eigenvalue regularisation for batch covariance.
   * - ``verbose``
     - bool
     - ``False``
     - Print progress information.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import TGLADDetector

   detector = TGLADDetector(window_size=256, stride=64, threshold=0.4)
   change_points = detector.fit_predict(X)

**Implementation:** Re-coding, with NumPy, of the scoring in ``main.py`` of the
authors' repository (`Harshs27/tGLAD <https://github.com/Harshs27/tGLAD>`_),
around the uGLAD multitask solver of that repository, vendored in
``tsseg/algorithms/tglad/vendor/uGLAD/`` (imports made relative).  uGLAD is
under the uGLAD Non-Commercial License, which allows non-commercial use only; a
copy is in ``tsseg/algorithms/tglad/vendor/LICENSE``.  *Origin: wrapper around
vendored third-party code.*

**Reference:** Imani & Shrivastava (2023), *Are uGLAD?  Time will tell!*

API reference
-------------

.. automodule:: tsseg.algorithms.tglad.detector
   :members:
   :show-inheritance:
   :undoc-members:
