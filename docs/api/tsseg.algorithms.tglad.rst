tsseg.algorithms.tglad package
==============================

tGLAD — temporal Graph-difference change point detection.

Description
-----------

tGLAD segments multivariate time series by tracking the evolution of
conditional independence (CI) graphs over time.  A sliding window converts the
series into overlapping intervals; the uGLAD sparse graph recovery model
(multitask mode) recovers a precision matrix (CI graph) for each interval
simultaneously.  A second-order trajectory-tracking algorithm measures
graph-difference scores, and an allocation algorithm produces the final
segmentation.

| **Type:** change point detection
| **Supervision:** fully unsupervised
| **Scope:** univariate and multivariate
| **Requires:** PyTorch, networkx

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
     - Min Frobenius distance to trigger a change point.
   * - ``min_spacing``
     - int / None
     - ``None``
     - Min distance between emitted CPs (default: ``stride``).
   * - ``epochs``
     - int
     - ``2000``
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
   labels = detector.fit_predict(X)

**Implementation:** *Origin: new code.*  Inspired by the tGLAD paper.

**Reference:** Imani & Shrivastava (2023), *Are uGLAD?  Time will tell!*

Submodules
----------

tsseg.algorithms.tglad.detector module
--------------------------------------

.. automodule:: tsseg.algorithms.tglad.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.tglad
   :show-inheritance:
