FLUSS
=====

FLUSS – Fast Low-cost Unipotent Semantic Segmentation.

Description
-----------

FLUSS is a domain-agnostic method for semantic segmentation of time series.  It
leverages the *matrix profile* (computed by ``stumpy``) and derives the *arc
curve*: for each time index the number of nearest-neighbour arcs crossing that
index is counted.  Valleys in the arc curve indicate semantic boundaries because
few matching subsequences span across a true regime change.

A sequential peak search on the inverted arc curve yields the requested number
of change points, or fewer when the curve left outside the exclusion zones of
the first ones is flat.  FLUSS has only one parameter – the window size – which
corresponds to the dominant period length.

By default, as in the FLUSS paper, the matrix profile compares z-normalized
subsequences, that is their shapes: a change in the mean or the scale of the
series alone does not show.  On three segments of white noise with different
means, the nearest neighbour of a window lies in its own segment for a third
of the windows only, as by chance, and no change point stands out.

``normalize=False`` compares the raw subsequences (Euclidean distance), so a
shift in level shows: on the same series (change points at 300 and 700,
default ``window_size=10``), three segments give ``[293, 692]``.  The arcs then stay in their segment and
the corrected arc curve is low over whole segments, not only at the change
points: without ``n_segments``, the default ``threshold=0.5`` returns twelve
change points there, ``threshold=0.1`` the two true ones.

On a multivariate series, ``multivariate_strategy`` chooses how the channels
are combined: ``"ensembling"`` (default) runs FLUSS on each channel and votes
on their change points, ``"l2"`` runs it on the L2 norm of the channels, and
``"cac_mean"`` averages the corrected arc curves of the channels, then
extracts the change points from that mean curve as from a univariate one.
The mean is the multivariate FLUSS of Gharghabi et al. (2019, section 3.4).
Each curve is already in [0, 1], so no rescaling is applied; ``profile``
holds the mean curve. A change seen by a few channels only is diluted by the
others, and ``threshold`` keeps its univariate default, not tuned for the
mean curve.

| **Type:** change point detection
| **Supervision:** semi-supervised or unsupervised
| **Scope:** univariate (multivariate via ensembling, the L2 norm or the
  mean corrected arc curve)
| **Complexity:** :math:`O(d\,n^{2})`: one matrix profile per channel
  (``stumpy.stump``)
| **Requires:** stumpy (``pip install tsseg[fluss]``)

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
     - ``10``
     - Sliding window size (use the dominant period length).
   * - ``n_segments``
     - int / None
     - ``2``
     - Number of segments (= change points + 1), at most. ``None``: unknown,
       change points are the minima of the corrected arc curve below
       ``threshold``.
   * - ``exclusion_factor``
     - int
     - ``5``
     - Multiplying factor for the exclusion zone around detected points.
   * - ``multivariate_strategy``
     - str
     - ``"ensembling"``
     - Strategy for multivariate inputs: ``"ensembling"`` (vote on the
       change points of the channels), ``"l2"`` (L2 norm of the channels) or
       ``"cac_mean"`` (mean of the corrected arc curves of the channels).
   * - ``tolerance``
     - float
     - ``0.01``
     - Ensembling: tolerance for aggregating change points (a float below 1
       is a fraction of the series length).
   * - ``threshold``
     - float
     - ``0.5``
     - With ``n_segments=None``: corrected-arc-curve value below which a
       minimum is a change point (in [0, 1]).
   * - ``consensus``
     - float
     - ``0.0``
     - Ensembling with ``n_segments=None``: detections of the channels within
       ``tolerance`` are merged; a merged change point is returned if at
       least this fraction of the channels (one at least) detected it.
   * - ``normalize``
     - bool
     - ``True``
     - ``True``: matrix profile of z-normalized subsequences (FLUSS paper).
       ``False``: raw subsequences, Euclidean distance (shifts in level
       show).
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import FLUSSDetector

   detector = FLUSSDetector(window_size=50, n_segments=4)
   change_points = detector.fit_predict(X)

**Implementation:** Wrapper around ``stumpy``.  BSD 3-Clause.

**Reference:** Gharghabi, Yeh, Ding, Ding, Hibbing, LaMunion, Kaplan, Crouter &
Keogh (2017, 2019), *Domain Agnostic Online Semantic Segmentation*, ICDM / DMKD.

API reference
-------------

.. automodule:: tsseg.algorithms.fluss.detector
   :members:
   :show-inheritance:
   :undoc-members:
