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

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
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
     - ``100``
     - uGLAD training epochs per batch.
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
