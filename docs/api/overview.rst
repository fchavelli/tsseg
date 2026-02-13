.. _api-overview:

Core API overview
=================

The :mod:`tsseg` package provides a thin façade over several namespaces that
are described throughout the guides. This page recaps the most relevant entry
points and links to the narrative documentation where the full usage examples
live.

Segments and detectors
----------------------

``tsseg.algorithms`` exposes detector classes for two main tasks:

**Change point detection** — AMOC, BinSeg, BOCD, BottomUp, ClaSP, DynP,
E-Agglo, ESPRESSO, FLUSS, GGS, iCID, IGTS, KCPD, PELT, Prophet, TIRE,
TS-CP2, tGLAD, Window.

**State detection** — AutoPlait, CLaP, E2USD, HDP-HSMM, Hidalgo, HMM,
PaTSS, TICC, Time2State, VQ-TSS, VSAX.

**Both (via TiRex)** — TiRex bridge module (under development).

A *Random* baseline is also included as a lower bound for benchmarks.

Each detector follows the standard ``fit``/``predict``/``fit_predict``
contract documented in :ref:`guides-detectors`.

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
