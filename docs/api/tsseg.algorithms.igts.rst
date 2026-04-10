tsseg.algorithms.igts package
=============================

IGTS — Information Gain based Temporal Segmentation.

Description
-----------

IGTS is a top-down greedy algorithm that locates the change point maximising the
information gain at each step, then repeats on the resulting sub-signals.  It
works best on multivariate series where distribution shifts across channels
provide discriminative evidence.

.. warning::
   IGTS does not perform well on univariate series without augmentation.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised
| **Scope:** multivariate (primarily)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``k_max``
     - int
     - ``10``
     - Maximum number of change points.
   * - ``step``
     - int
     - ``5``
     - Stride for candidate locations.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import InformationGainDetector

   detector = InformationGainDetector(k_max=8, step=5)
   labels = detector.fit_predict(X)

**Implementation:** Adapted from aeon.  BSD 3-Clause.

**Reference:** Sadri, Ren & Salim (2017), *Information Gain-based Metric for
Recognizing Transitions in Human Activities*, Pervasive and Mobile Computing.

Submodules
----------

tsseg.algorithms.igts.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.igts.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.igts
   :show-inheritance:
