.. _changelog:

Changelog
=========

This changelog highlights notable updates. For full commit history, refer to
`GitHub <https://github.com/fchavelli/tsseg/commits/main>`_.

Unreleased
----------

* Vendored a lightweight subset of ``ruptures`` and migrated BinSeg, BottomUp,
  Dynp, and PELT detectors to use it (no external dependency required).
* Added BottomUp and Dynp detector wrappers with aeon-compatible interfaces.
* Replaced the surrogate PyTorch ``TSCP2Detector`` with a faithful
    TensorFlow + TCN implementation matching the original baseline.
* Replaced the ``pyhsmm``-backed ``HdpHsmmDetector`` with a native
    NumPy/SciPy sticky HDP implementation and preserved a legacy wrapper for
    reference.
* Added Sphinx documentation scaffold with GitHub Pages deployment.
* Documented test suite layout and fixtures.
* Improved detector axis handling for ``ClaspDetector``.

Earlier releases
----------------

(Backfill entries here as tags or releases are published.)
