Prophet
=======

Prophet — trend change point detection via Facebook Prophet.

Description
-----------

This detector returns the change points of Facebook Prophet's piecewise-linear
trend.  Prophet lets the trend's rate change at a set of *potential* change
points, with an L1-regularised (sparse, Laplace) prior on the rate-change
magnitudes ``delta`` so that most potential points go unused.  The detector
places ``n_candidates`` potential change points evenly over the series and
scores each one by its fitted ``|delta|`` relative to the median ``|delta|``
over the candidates.  It returns the ``n_changepoints`` strongest when this
number is given, otherwise those whose score exceeds ``delta_threshold`` (a
relative form of TCPDBench's rule).  Two returned change
points are at least ``tolerance`` apart, since a level shift between two
candidates shows as two opposite rate changes on consecutive candidates.

Key parameters:

- ``n_changepoints`` — number of change points to return (semi-supervised);
  ``None`` thresholds the rate changes instead.
- ``n_candidates`` — number of evenly spaced potential change points; their
  spacing bounds the localisation accuracy.
- ``changepoint_prior_scale`` — scale of the Laplace prior: larger values
  give sharper trend changes, closer to the true change points.  The default
  (500) was calibrated for change point detection on TSB-SEG, far above
  Prophet's forecasting default (0.05).
- ``max_points`` — longer series are reduced to block means before the fit.

| **Type:** change point detection
| **Supervision:** semi-supervised (``n_changepoints``) or unsupervised
| **Scope:** univariate (multivariate via ensembling or L2 norm)
| **Complexity:** :math:`O(d\,n)` for the L2 reduction (default strategy), then
  :math:`O(\tilde{n}\,S)` per L-BFGS iteration of Prophet's fit, where
  :math:`\tilde{n} = \min(n, 2000)` (``max_points``) and *S* is
  ``n_candidates``
| **Requires:** ``prophet`` and ``cmdstanpy``

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 22 12 10 56

   * - Name
     - Type
     - Default
     - Description
   * - ``n_changepoints``
     - int / None
     - ``None``
     - Number of change points to return; ``None`` selects them by
       ``delta_threshold``.
   * - ``n_changepoint_func``
     - callable / None
     - ``None``
     - Callable that determines ``n_changepoints`` from the series.
   * - ``n_candidates``
     - int
     - ``200``
     - Number of evenly spaced potential change points.
   * - ``changepoint_prior_scale``
     - float
     - ``500.0``
     - Scale of the Laplace prior on the rate changes.
   * - ``delta_threshold``
     - float
     - ``3.0``
     - Minimum score (``|delta|`` over its median) of a change point when
       ``n_changepoints`` is ``None``.
   * - ``seasonality``
     - bool
     - ``False``
     - Fit Prophet's yearly, weekly and daily seasonalities.
   * - ``multivariate_strategy``
     - str
     - ``"l2"``
     - ``"ensembling"`` (one model per channel, scores averaged over the
       channels) or ``"l2"`` (one model on the L2 norm).
   * - ``tolerance``
     - float
     - ``0.02``
     - Minimum distance between two change points (fraction of the length
       when below 1, samples otherwise).
   * - ``max_points``
     - int / None
     - ``2000``
     - Series longer than this are reduced to block means before the fit.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ProphetDetector

   detector = ProphetDetector(n_changepoints=10)
   change_points = detector.fit_predict(X)

**Implementation:** Wrapper around ``facebook/prophet``.  MIT.

**Reference:** Taylor & Letham (2018), *Forecasting at Scale*, The American
Statistician.

API reference
-------------

.. automodule:: tsseg.algorithms.prophet.detector
   :members:
   :show-inheritance:
   :undoc-members:
