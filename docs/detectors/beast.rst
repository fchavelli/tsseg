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
       The last three can crash Rbeast 0.1.25 (see below).
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
     - Seed of the MCMC sampler (``0``: random seed, not reproducible). A
       fixed seed reproduces the output on one machine only (see below).
   * - ``cp_prob_threshold``
     - float
     - ``0.1``
     - Minimum posterior probability to accept a change point.
   * - ``max_cps``
     - int / None
     - ``None``
     - Optional upper bound on the number of change points (the most
       probable ones are kept).
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
     - Distance (time steps; a float below 1 is a fraction of the length)
       within which close change points count as one, with ``"native"``
       (the most probable is kept) and ``"ensembling"`` (median of the
       group). ``0``: no merge beyond identical time steps and, with
       ``"native"``, candidates of different channels closer than Rbeast's
       minimum distance between two change points of one channel.
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

**Reproducibility:** Rbeast picks its numerical routines by CPU instruction
set (SSE, AVX2, AVX-512), and the MCMC sampler amplifies their rounding
differences, so a fixed ``mcmc_seed`` gives the same output on one machine,
not necessarily on another (`Rbeast issue #53
<https://github.com/zhaokg/Rbeast/issues/53>`_). On some machines, Rbeast
0.1.25 returned NaN for every output on a 6,000-point series in which it
finds the change point elsewhere. `Rbeast issue #56
<https://github.com/zhaokg/Rbeast/issues/56>`_ reports wrong outputs on Intel
Xeon CPUs. When the median number of change points (``ncp_median``) is NaN in
every component and every channel, the detector raises a ``RuntimeWarning``
and reruns Rbeast once with its SSE routines (``cputype="sse"``, which took
1.5 to 2 times as long on an AVX2 CPU); if the rerun returns NaN too, the
detector returns no change point. Other runs are unchanged.

**Crashes with seasonal models:** with ``season`` set to ``"harmonic"``,
``"dummy"`` or ``"svd"``, Rbeast 0.1.25 can kill the Python process (abort on
a double free or a heap corruption, or segmentation fault) instead of raising
an error. A direct call to Rbeast crashes the same way. In about 3,300 calls,
each in a fresh process, whether a call crashed depended on the length of the
series, the period and the values, so the parameters alone do not tell which
calls are safe. On univariate series:

- ``"dummy"`` with ``period=2`` crashed on every series of 3 to 5 points and
  on none of 6 to 8 points. With ``period=12``, it crashed on 200 points of
  white noise, but not on a sine of period 12 plus noise, a trend or a step
  of the same length.
- ``"harmonic"`` with ``period=12`` crashed on most white-noise series of 18
  to 25 points, and on none of 16, 17, 26 or 27 points.
- ``"svd"`` with ``period=4`` crashed on some series of 3 to 14 points.
- A few calls returned a result, and the process died later.
- ``season="none"`` (the default) never crashed.

To keep the caller alive, run each call in a new child process. A crash then
only kills the child, and ``concurrent.futures`` raises ``BrokenProcessPool``
in the caller:

.. code-block:: python

   from concurrent.futures import ProcessPoolExecutor
   from concurrent.futures.process import BrokenProcessPool

   from tsseg.algorithms import BeastDetector


   def detect(X):
       return BeastDetector(season="dummy", period=12).fit_predict(X)


   if __name__ == "__main__":
       with ProcessPoolExecutor(max_workers=1) as pool:  # one pool per call
           try:
               change_points = pool.submit(detect, X).result()
           except BrokenProcessPool:
               change_points = None  # Rbeast killed the child process

**Reference:** Zhao et al. (2019), *Detecting change-point, trend, and
seasonality in satellite time series data*, Remote Sensing of Environment.
