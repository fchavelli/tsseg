BEAST
=====

BEAST (Bayesian Estimator of Abrupt change, Seasonality, and Trend).

Description
-----------

BEAST decomposes a time series into trend + seasonality + noise via Bayesian
model averaging (BMA) over a large space of piecewise-linear trend models and
piecewise-harmonic seasonal models.  Change points are inferred from the
posterior distribution using Reverse-Jump MCMC (RJ-MCMC) and Gibbs sampling.
Each time step receives a posterior probability of being a change point.

Instead of selecting a single "best" segmentation, BEAST integrates over all
candidate models weighted by their posterior probability, yielding robust
detection with uncertainty quantification.

Three strategies handle multivariate inputs:

- ``"native"`` — uses ``beast123`` which handles multi-dimensional data natively.
- ``"l2"`` — reduces channels via L2 norm before inference.
- ``"ensembling"`` — runs BEAST per channel and aggregates change points.

| **Type:** change point detection
| **Supervision:** unsupervised or semi-supervised (``max_cps``)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(d\,n)` per MCMC iteration
| **Requires:** Rbeast >= 0.1.25 (``pip install tsseg[beast]``)

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 60

   * - Name
     - Type
     - Default
     - Description
   * - ``season``
     - str
     - ``"none"``
     - Seasonal model type: ``"none"``, ``"harmonic"``, ``"dummy"``, ``"svd"``.
   * - ``tcp_minmax``
     - tuple
     - ``(0, 10)``
     - (min, max) number of trend change points.
   * - ``torder_minmax``
     - tuple
     - ``(0, 1)``
     - (min, max) polynomial orders for the trend.
   * - ``tseg_minlength``
     - int / None
     - ``None``
     - Minimum segment length for the trend component.
   * - ``scp_minmax``
     - tuple
     - ``(0, 10)``
     - (min, max) number of seasonal change points.
   * - ``sorder_minmax``
     - tuple
     - ``(0, 5)``
     - (min, max) harmonic orders for the seasonal component.
   * - ``sseg_minlength``
     - int / None
     - ``None``
     - Minimum segment length for the seasonal component.
   * - ``period``
     - float
     - ``NaN``
     - Period of the seasonal signal (e.g. 12 for monthly/annual).
   * - ``deltat``
     - float
     - ``1.0``
     - Time interval between consecutive observations.
   * - ``mcmc_samples``
     - int
     - ``8000``
     - Number of MCMC samples to collect.
   * - ``mcmc_burnin``
     - int
     - ``200``
     - Number of initial samples to discard per chain.
   * - ``mcmc_chains``
     - int
     - ``3``
     - Number of MCMC chains to run.
   * - ``mcmc_thin``
     - int
     - ``5``
     - Thinning factor for the MCMC chains.
   * - ``mcmc_seed``
     - int
     - ``1``
     - Seed of the MCMC sampler (``0``: random seed, not reproducible).
   * - ``cp_prob_threshold``
     - float
     - ``0.1``
     - Minimum posterior probability to accept a change point.
   * - ``max_cps``
     - int / None
     - ``None``
     - Optional upper bound on the number of change points.
   * - ``component``
     - str
     - ``"trend"``
     - Which component's change points to return: ``"trend"``, ``"season"``, ``"both"``.
   * - ``multivariate_strategy``
     - str
     - ``"native"``
     - ``"native"`` (beast123), ``"l2"`` (L2 norm), or ``"ensembling"`` (per-channel).
   * - ``tolerance``
     - float
     - ``0``
     - Tolerance for aggregating change points in ensembling mode.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import BeastDetector

   # Trend-only detection (no seasonality)
   detector = BeastDetector(season="none", cp_prob_threshold=0.2)
   change_points = detector.fit_predict(X)

   # Seasonal data (e.g. monthly with annual period)
   detector = BeastDetector(season="harmonic", period=12)
   change_points = detector.fit_predict(X)

**Implementation:** wraps the `Rbeast <https://github.com/zhaokg/Rbeast>`_
package, version 0.1.25 or later (``pip install tsseg[beast]``).

**Reference:** Zhao et al. (2019), *Detecting change-point, trend, and
seasonality in satellite time series data*, Remote Sensing of Environment.
