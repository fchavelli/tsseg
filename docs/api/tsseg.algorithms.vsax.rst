tsseg.algorithms.vsax package
=============================

VSAX — Variable-length SAX state detection.

Description
-----------

VSAX converts each channel to Symbolic Aggregate approXimation (SAX) symbols,
then finds the variable-length segmentation that minimises PAA reconstruction
error plus an additive penalty per segment.  The pipeline:

1. **Z-normalisation** — optionally standardise each channel.
2. **PAA** — reduce each candidate segment to ``paa_segments`` frames.
3. **SAX** — discretise PAA values into ``alphabet_size`` symbols using
   Gaussian breakpoints (or adaptive empirical quantiles).
4. **DP segmentation** — dynamic programming over ``num_lengths`` candidate
   segment lengths minimises reconstruction error + ``penalty`` per segment.
5. **Symbol merging** — per-channel SAX symbol tuples are clustered via
   agglomerative clustering on Hamming distance (threshold
   ``symbol_merge_threshold``).

| **Type:** state detection
| **Supervision:** semi-supervised or unsupervised
| **Scope:** univariate

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 55

   * - Name
     - Type
     - Default
     - Description
   * - ``alphabet_size``
     - int
     - ``6``
     - Number of SAX symbols per channel.
   * - ``paa_segments``
     - int
     - ``8``
     - Number of PAA frames per segment.
   * - ``min_segment_length``
     - int
     - ``20``
     - Minimum admissible segment length.
   * - ``max_segment_length``
     - int
     - ``180``
     - Maximum admissible segment length.
   * - ``num_lengths``
     - int
     - ``6``
     - Number of candidate lengths (linearly spaced min..max).
   * - ``penalty``
     - float
     - ``0.8``
     - Cost per new segment.  Larger values produce longer segments.
   * - ``symbol_merge_threshold``
     - float
     - ``0.2``
     - Normalised Hamming distance threshold for merging symbols.
       ``0`` = exact match only, ``1`` = single global state.
   * - ``zscore``
     - bool
     - ``True``
     - Apply per-channel z-normalisation.
   * - ``adaptive_breakpoints``
     - bool
     - ``True``
     - Learn SAX breakpoints from empirical quantiles.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import VSAXDetector

   detector = VSAXDetector(
       alphabet_size=8, penalty=1.0, min_segment_length=30) states =
   detector.fit_predict(X)

**Implementation:** *Origin: new code.*

**Reference:** —

Submodules
----------

tsseg.algorithms.vsax.detector module
-------------------------------------

.. automodule:: tsseg.algorithms.vsax.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.vsax
   :show-inheritance:
