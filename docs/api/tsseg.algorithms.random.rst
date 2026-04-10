tsseg.algorithms.random package
===============================

Random — baseline detector for benchmarking.

Description
-----------

Emits uniformly random change points or state labels.  Intended as a lower
bound for benchmarks and for pipeline testing.

Two modes of operation:

- **Unsupervised** — draws a random ``n_change_points`` and/or ``n_states``.
- **Semi-supervised** — given quantities, randomises locations/labels.

| **Type:** state detection
| **Supervision:** unsupervised or semi-supervised
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
   * - ``semi_supervised``
     - bool
     - ``False``
     - Enable semi-supervised mode.
   * - ``n_change_points``
     - int / None
     - ``None``
     - Number of change points to emit.
   * - ``n_states``
     - int / None
     - ``None``
     - Number of distinct states.
   * - ``random_state``
     - int / None
     - ``None``
     - Random seed for reproducibility.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import RandomDetector

   detector = RandomDetector(n_change_points=5, random_state=42)
   labels = detector.fit_predict(X)

**Implementation:** *Origin: new code.*

**Reference:** —

Submodules
----------

tsseg.algorithms.random.detector module
---------------------------------------

.. automodule:: tsseg.algorithms.random.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.random
   :show-inheritance:
