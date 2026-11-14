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

- ``"native"`` – runs ``beast123`` on each channel (Rbeast fits the channels
  independently) and pools their candidates: a change seen by several
  channels comes first, and without ``max_cps`` as many are returned as a
  channel that changes has (see below).
- ``"l2"`` – reduces channels via L2 norm before inference.
- ``"ensembling"`` – runs BEAST per channel and aggregates change points.

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
     - (min, max) number of trend change points, per channel. A larger
       ``max_cps`` raises the max.
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
       probable ones are kept). Above the max of ``tcp_minmax`` (or of
       ``scp_minmax``, for a seasonal component returned), it raises that
       max: Rbeast never returns more change points per component and
       channel.
   * - ``component``
     - str
     - ``"trend"``
     - Which component's change points to return: ``"trend"``, ``"season"``, ``"both"``.
   * - ``multivariate_strategy``
     - str
     - ``"native"``
     - ``"native"`` (beast123 per channel), ``"l2"`` (L2 norm), or ``"ensembling"`` (per-channel).
   * - ``tolerance``
     - float
     - ``0``
     - Distance (time steps; a float below 1 is a fraction of the length)
       within which close change points count as one, with ``"native"``
       (the most supported is kept) and ``"ensembling"`` (median of the
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
package, version 0.1.25 or later (``pip install tsseg[beast]``). On Python 3.13
and later, Rbeast 0.1.25 prints ``Exception ignored in
PyObject_HasAttrString()`` and a ``SystemError`` to standard error at each call:
its C extension misuses a function of the Python C API, which Python 3.12 and
earlier ignore silently. The change points are unaffected.

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
detector returns no change point. Other runs are unchanged. In the
continuous integration of tsseg, the first run returned NaN on that series in
10 of 177 test jobs (27 October to 6 November 2026), all on CPUs with
AVX-512, and the rerun found the change point every time.

**Pooling of the channels (``"native"``):** each channel gives its own
candidates, up to the max of ``tcp_minmax`` (10 by default). On real
multivariate series, the trend model of BEAST fits most channels with that
many: on the first 2,000 points of 435 real series from 47 datasets
(z-normalised channels), 74 % of the channels had a median number of change
points of 10. Their union, returned until tsseg 0.3.2, thus grew
with the number of channels: 15 times the true number of change points in the
median multivariate series, scores near those of a random segmentation.
Candidates are now taken by support, the sum over the channels of the
probability (``cpPr``) of the candidate of each channel at the same step
(closer than Rbeast's minimum distance, or within ``tolerance``), and without
``max_cps`` their number is the median number of candidates of the channels
that have any. The count assumes changes shared by the channels: when
channels change at different times, some of their changes are left out.

**Cluster jobs and containers:** given a 2D array, Rbeast 0.1.25 fits its
columns independently, in threads pinned to CPUs 0, 1, ... whatever CPUs the
process may use. In a process that may not use these CPUs (a job of a cluster
scheduler that allots it other CPUs, a container limited to some CPUs), the
threads are not created: Rbeast returns unset outputs and corrupts the
memory, so that the process is killed later. The ``"native"`` strategy
therefore runs ``beast123`` on one channel at a time, which Rbeast computes
in the calling thread; the channels are fitted one after the other, not in
parallel. Where the threads could be created, its outputs can differ from
those of the run on the whole array, which goes through other routines of
Rbeast.

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
