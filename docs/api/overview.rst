.. _api-overview:

Core API overview
=================

The :mod:`tsseg` package provides a thin façade over several namespaces that
are described throughout the guides. This page recaps the most relevant entry
points and links to the narrative documentation where the full usage examples
live.

Segments and detectors
----------------------

`tsseg.algorithms` exposes detector classes such as
:class:`~tsseg.algorithms.EspressoDetector` or
:class:`~tsseg.algorithms.PatssDetector`. Each detector follows the standard
``fit``/``predict``/``fit_predict`` contract documented in
:ref:`guides-detectors`. The module also re-exports helpers like
:class:`~tsseg.algorithms.Time2StateDetector` for state-labelling tasks.

Datasets
--------

`tsseg.datasets` bundles a small collection of toy datasets plus convenience
functions for loading datasets from disk. Refer to
:doc:`../guides/datasets` for details on file formats and helper functions.

Segmentation primitives
-----------------------

`tsseg.segment` defines light-weight structures describing change points and
windows. They are primarily used internally but can be helpful when wiring the
library into custom pipelines.

Evaluation
----------

`tsseg.metrics` contains scoring helpers for both change-point detection and
state-labelling evaluation. The most common utilities are summarised in
:doc:`metrics`.

Utilities
---------

Supporting tools such as configuration helpers and the testing fixtures live in
`tsseg.utils`. They are considered internal but can occasionally simplify
integration scenarios.
