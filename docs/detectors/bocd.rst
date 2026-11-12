BOCD
====

Offline Bayesian Gaussian segmentation with joint MAP decoding.

Description
-----------

``BOCDDetector`` fits the Gaussian product-partition model of Fearnhead (2006).
A constant hazard :math:`h=1/\lambda` gives segment duration mass
:math:`g(l)=h(1-h)^{l-1}`; the final segment contributes its right-censored
survival probability. The default returns the single most probable partition
(MAP), using the recursion of Fearnhead and Liu (2007). The whole series is
used. The historical name does not imply the causal online filter of Adams
and MacKay (2007).

The default ``multivariate_strategy="native"`` integrates out the mean vector
and full covariance under a Normal-inverse-Wishart prior. It can distinguish
changes of direction or covariance that a norm reduction loses. For
:math:`d` channels, the prior parameters are

.. math::

   \mu_0 = \mu\mathbf{1}, \qquad \kappa_0=\kappa, \qquad
   \nu_0 = 2\alpha+d-1, \qquad \Psi_0 = 2\beta I_d.

Each marginal variance has an inverse-gamma prior with shape ``alpha`` and
scale ``beta``. For one channel this is exactly the Normal-Gamma model.
The alternatives ``"l2"`` and ``"ensembling"`` respectively analyse the norm or
fit each channel separately and merge the boundaries. Ensembling does not
model cross-channel covariance; MAP then applies to each channel separately.

| **Type:** change point detection
| **Supervision:** unsupervised or an upper bound on the number of changes
| **Scope:** univariate and multivariate

Defaults and decision rule
--------------------------

The defaults use per-channel standardisation, ``hazard_lambda=1000``,
``alpha=kappa=1``, ``beta=10``, a minimum segment length of 25, and joint MAP
inference on at most 2,000 interior candidate positions. These values were
selected on 15 series from five benchmark datasets, with basic mean, variance,
covariance and stationary-noise checks. They are a practical starting point,
not a guarantee of suitability for every time scale or observation model.

``standardize=True`` centers and scales the series supplied to ``predict``;
prior parameters are in these standardised units. In L2 mode, the norm is
computed first and its scalar series is then standardised, so that centering
channels does not itself erase a magnitude change. To supply priors in physical
units, set ``standardize=False``.

``max_cps=K`` constrains MAP inference to partitions with **at most** K changes.
It optimises their locations jointly; it does not trim a previously fitted
partition and does not force exactly K changes. ``max_cps=0`` skips inference.
``min_distance`` constrains segment lengths during inference, including both
edge segments. Ensembling also spaces its final aggregated boundaries.

For marginal rather than MAP decoding, set ``decoding="marginal"``.
``cp_prob_threshold=0.5`` then thresholds peaks of the boundary probabilities,
with spacing and an optional count cap. Boundary probabilities are distinct
events: their sum is the posterior expected number of changes, not one.
The probability threshold and ``truncate`` do not affect MAP decoding.
``truncate=None`` sums every message term. The optional finite legacy cutoff
can discard important later terms; it does not accelerate likelihood evaluation.

Candidate grid and cost
-----------------------

``max_candidates=2000`` restricts the possible boundary locations on a regular
grid, always including 0 and n as segment endpoints. **Every observation** still
contributes to segment means and covariance matrices. This is a restriction on
boundary locations, not downsampling or replacing blocks by their means. Both
MAP and marginal inference are conditional on this candidate set and the minimum
segment length. Set ``max_candidates=None`` to allow every interior position.
The geometric prior retains the original sample-time units on this grid;
changing the grid changes the conditioned model.

Let :math:`m` be the number of candidate vertices, including the two endpoints.
For native multivariate inference, worst-case time is
:math:`O(nd^2+m^2d^3)`, with :math:`O(n+md^2)` working memory beyond the input.
For one channel it is :math:`O(n+m^2)` time and :math:`O(n+m)` memory. An active
MAP count constraint K adds :math:`O(m^2K)` time and :math:`O(mK)` memory;
unconstrained MAP needs no table indexed by the number of changes. Ensembling
runs the univariate calculation per channel and stores their candidates.

Block moments are combined using Welford's recurrence. A determinant lemma
keeps large mean offsets out of covariance matrices; Cholesky evaluation falls
back to an eigenvalue calculation if a tiny prior ridge is not numerically
resolved. No n-by-n segment or ordinal-change matrix is stored.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import BOCDDetector

   change_points = BOCDDetector().fit_predict(X)
   at_most_three = BOCDDetector(max_cps=3).fit_predict(X)
   probabilities_as_peaks = BOCDDetector(
       decoding="marginal", cp_prob_threshold=0.5
   ).fit_predict(X)

Suitability and earlier results
-------------------------------

The model assumes independent Gaussian observations within a segment. Full
covariance detects more multivariate distribution changes than L2 reduction,
but does not describe arbitrary temporal dynamics. Oscillations, drift or
outliers can still produce extra boundaries; short weak changes can be missed.
``hazard_lambda`` and ``min_distance`` are in samples and depend on the sampling
rate. Check both boundary precision and segmentation quality for the application.

Versions before the inference correction passed a constant duration mass,
normalised boundary scores to sum to one, and used cubic-time inference with
quadratic memory. The corrected inference first retained the old defaults.
The new native likelihood, MAP decision rule, standardisation, candidate grid
and calibrated defaults change the outputs again. Earlier BOCD benchmark
results must be recomputed for this configuration.

When migrating an experiment configuration, explicit old values still override
the constructor defaults. Set ``hazard_lambda=1000`` and ``beta=10`` (or remove
their overrides) to use the calibrated prior. In MAP mode, remove probability
threshold and truncation axes from tuning grids: they are inactive; vary the
duration and Gaussian prior parameters instead. For example:

.. code-block:: yaml

   instance:
     _target_: tsseg.algorithms.bocd.detector.BOCDDetector
     multivariate_strategy: native
     decoding: map
     hazard_lambda: 1000
     beta: 10
     standardize: true
     max_candidates: 2000
     truncate: null

**Implementation:** the detector uses ``_offline.py``. Legacy generic helpers
in ``bayesian_models.py``, originally adapted from hildensia under MIT, remain
available for compatibility and are not called by the detector.

**References:** `Fearnhead (2006)
<https://eprints.lancs.ac.uk/id/eprint/8189/>`_; `Fearnhead and Liu (2007),
Section 3.1 <https://eprints.lancs.ac.uk/id/eprint/745/>`_; `Murphy (2007),
conjugate Gaussian analysis <https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf>`_.

API reference
-------------

.. automodule:: tsseg.algorithms.bocd.detector
   :members:
   :show-inheritance:
   :undoc-members:

.. automodule:: tsseg.algorithms.bocd.bayesian_models
   :members:
   :show-inheritance:
   :undoc-members:
