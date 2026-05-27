Detectors
=========

``tsseg.algorithms`` exposes 30+ segmentation algorithms behind a common
``fit`` / ``predict`` / ``fit_predict`` API derived from
:class:`aeon.segmentation.base.BaseSegmenter`.

Each detector declares its capabilities through tags:

* ``capability:univariate`` / ``capability:multivariate``
* ``returns_dense`` — ``True`` for state labels, ``False`` for sparse change
  points
* ``semi_supervised`` — ``True`` when the algorithm accepts labels at fit time

Use the table of contents below to jump to a specific detector. The detectors
are grouped into two tasks: **change-point detection** returns sparse breakpoint
indices, **state detection** returns dense state labels.

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
   hdp_hsmm
   hidalgo
   hmm
   patss
   ticc
   time2state
   vqtss
   vsax

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
   ``n_states``) must now be passed explicitly to the constructor.

See :doc:`../guides/contributing` for the full detector-side contract (tags,
output shapes and the estimator-check test suite that every algorithm in
``__all__`` must pass).
