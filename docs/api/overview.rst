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

:mod:`tsseg.data.datasets` bundles a small collection of datasets (e.g. MoCap)
plus convenience functions for loading data. Refer to
:doc:`../guides/datasets` for details on file formats and helper functions.

Evaluation
----------

:mod:`tsseg.metrics` contains scoring helpers for both change-point detection and
state-labelling evaluation. The most common utilities are summarised in
:doc:`metrics`.
