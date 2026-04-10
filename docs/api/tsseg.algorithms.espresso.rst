tsseg.algorithms.espresso package
=================================

ESPRESSO — Entropy and Shape-aware Time Series Segmentation.

Description
-----------

ESPRESSO exploits both entropy and temporal shape properties to segment
multi-dimensional time series.  The method builds a *semantic density matrix*
from the matrix profile using an arc-curve expansion procedure:

1. Compute the matrix profile with ``stumpy``.
2. Iteratively expand arc sets (controlled by ``chain_len``) to accumulate
   segment-boundary evidence.
3. Detect peaks in the resulting density curve; peaks signal change points.

ESPRESSO differs from methods that focus exclusively on statistical or temporal
properties.

| **Type:** change point detection
| **Supervision:** semi-supervised (``n_segments`` required)
| **Scope:** univariate and multivariate

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
     - ``64``
     - Subsequence length for the matrix profile (>= 4).
   * - ``chain_len``
     - int
     - ``3``
     - Iterations for expanding arc sets.
   * - ``n_segments``
     - int / None
     - ``None``
     - Target number of segments (>= 2 at predict time).
   * - ``peak_distance_fraction``
     - float
     - ``0.01``
     - Minimum spacing between peaks as a fraction of the series length.
   * - ``axis``
     - int
     - ``0``
     - Time axis.
   * - ``random_state``
     - int / None
     - ``None``
     - Seed for the internal RNG.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import EspressoDetector

   detector = EspressoDetector(window_size=64, chain_len=5, n_segments=4)
   labels = detector.fit_predict(X)

**Implementation:** Reimplemented from the original ESPRESSO paper.  *Origin:
new code.*

**Reference:** Deldari, Smith, Sadri & Salim (2020), *ESPRESSO: Entropy and
ShaPe awaRe timE-Series SegmentatiOn*, UbiComp.

Submodules
----------

tsseg.algorithms.espresso.detector module
-----------------------------------------

.. automodule:: tsseg.algorithms.espresso.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.espresso
   :show-inheritance:
