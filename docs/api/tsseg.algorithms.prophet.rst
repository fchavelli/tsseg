tsseg.algorithms.prophet package
================================

Prophet — trend change point detection via Facebook Prophet.

Description
-----------

This detector wraps Facebook Prophet's piecewise-linear trend model.  Prophet
specifies a large number of *potential* change points uniformly placed in the
first 80 % of the time series, then applies an L1-regularised (sparse) prior on
the rate-change magnitudes so that most potential points go unused.  The
detector extracts the locations of the *significant* rate changes from the
fitted model.

Key parameters:

- ``n_changepoints`` — number of *potential* change points.
- ``changepoint_prior_scale`` (tunable inside ``cost_params``) — controls trend
  flexibility (default 0.05; increase for a more flexible trend).
- ``changepoint_range`` (tunable inside ``cost_params``) — fraction of the
  history where change points are allowed (default 0.8).

| **Type:** change point detection
| **Supervision:** semi-supervised (``n_changepoints`` recommended)
| **Scope:** univariate (multivariate via ensembling)
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
     - ``5``
     - Number of potential change points.
   * - ``n_changepoint_func``
     - callable / None
     - ``None``
     - Callable that determines ``n_changepoints`` from the series.
   * - ``multivariate_strategy``
     - str
     - ``"ensembling"``
     - Strategy for multivariate series (``"ensembling"`` or ``"l2"``).
   * - ``tolerance``
     - float
     - ``0.01``
     - Tolerance for change-point deduplication.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import ProphetDetector

   detector = ProphetDetector(n_changepoints=10)
   labels = detector.fit_predict(X)

**Implementation:** Wrapper around ``facebook/prophet``.  MIT.

**Reference:** Taylor & Letham (2018), *Forecasting at Scale*, The American
Statistician.

Submodules
----------

tsseg.algorithms.prophet.detector module
----------------------------------------

.. automodule:: tsseg.algorithms.prophet.detector
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: tsseg.algorithms.prophet
   :members:
   :show-inheritance:
   :undoc-members:
