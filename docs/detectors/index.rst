Detectors
=========

``tsseg.algorithms`` exposes 30+ segmentation algorithms behind a common
``fit`` / ``predict`` / ``fit_predict`` API. Every detector inherits from
``tsseg.algorithms.base.BaseSegmenter``, which follows the segmenter API of
`aeon <https://github.com/aeon-toolkit/aeon>`_ (a copy of it: ``tsseg`` does
not depend on aeon for the base class).

Each detector declares its capabilities through tags, read with
``detector.get_tag(name)``:

* ``detector_type`` — ``"change_point_detection"`` or ``"state_detection"``
* ``returns_dense`` — ``True`` when ``predict`` returns change point indices
  (aeon's "dense" format), ``False`` when it returns one state label per time
  point
* ``capability:univariate`` / ``capability:multivariate``
* ``capability:unsupervised`` — ``True`` when the detector runs without being
  told the number of change points or states
* ``capability:semi_supervised`` — ``True`` when that number can be passed to
  the constructor (guided mode, e.g. ``n_cps`` or ``n_states``)

Use the table of contents below to jump to a specific detector. The detectors
are grouped into two tasks: **change-point detection** returns a sorted array
of change point indices (each index is the first point of a new segment; by
convention ``0`` and the series length are left out), **state detection**
returns an array of integer state labels of the same length as the series.

Change-point detection
----------------------

.. toctree::
   :maxdepth: 1

   amoc
   beast
   binseg
   bocd
   bottomup
   changefinder
   dynp
   eagglo
   espresso
   fluss
   ggs
   icid
   igts
   kcpd
   pelt
   prophet
   ruptures
   seeded_binseg
   tglad
   tire
   tscp2
   window

State detection
---------------

.. toctree::
   :maxdepth: 1

   autoplait
   clap
   e2usd
   gaussian_hmm
   hdp_hsmm
   hidalgo
   hmm
   patss
   snlds
   ticc
   time2state

Baseline
--------

.. toctree::
   :maxdepth: 1

   random


Conventions
-----------

.. note::

   The ``fit`` signature is ``fit(X, y=None)``. For **unsupervised** algorithms
   ``y`` must be ``None``; passing it is reserved for **semi-supervised** or
   **supervised** detectors that explicitly support it.

   Parameters that used to be inferred from ``y`` (such as ``n_segments`` or
   ``n_states``) must now be passed explicitly to the constructor. Their name
   depends on the detector (``n_cps``, ``n_change_points``, ``n_segments``,
   ``n_states``, ...): see each detector's page.

See :doc:`../guides/contributing` for the full detector-side contract (tags,
output shapes and the estimator-check test suite that every algorithm in
``__all__`` must pass).
