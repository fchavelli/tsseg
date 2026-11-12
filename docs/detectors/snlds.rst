SNLDS
=====

Switching nonlinear dynamical systems fitted by collapsed variational inference.
A continuous latent trajectory is governed by a discrete regime sequence;
each regime has its own nonlinear continuous transition.

.. warning::

   **Experimental.** Numerical correctness and improved diagnostic scores do
   not establish reliable accuracy across datasets. The validation includes
   unsuccessful seeds and series. Current defaults are adaptations of SNLDS,
   not a reproduction of its published experiments.

| **Type:** state detection
| **Supervision:** semi-supervised (requires ``n_states``)
| **Scope:** univariate and multivariate, offline
| **Requires:** TensorFlow, TensorFlow Probability and ``tf-keras``
  (``pip install tsseg[snlds]``)

Model and training
------------------

The continuous state :math:`z_t` evolves under one of :math:`K` nonlinear
transitions, selected by :math:`s_t`. A shared network emits observations.
A bidirectional RNN embeds observations before inferring continuous states.
Discrete states are marginalized exactly by the forward recursion. The
collapsed ELBO uses its log normalizers, rather than an expected complete-data
score missing the discrete posterior entropy.

Gaussian scales are positive and learnable. Initial Gaussian means and
normalized initial regime logits are tracked by Keras. Gradients are clipped
by global norm. The pointwise uniform-state cross-entropy regularizer is
disabled by default: Appendix B.4.1 of
`Ansari et al., NeurIPS 2021 <https://arxiv.org/html/2110.13878v1>`_
describes the same exact-likelihood SNLDS correction without that regularizer.
No RED-SDS source code is copied.

The default continuous posterior is **factorized conditional on a bidirectional
observation embedding**. Its mean is temporal, but its variational noise is
independent over time, allowing latent draws to be evaluated at once.
``inference_type="autoregressive"`` retains a forward RNN additionally
conditioned on past latent draws, as in the SNLDS paper. Both are variational
families for the same generative model; their approximation quality can differ.

Initialization performs ``init_steps=200`` reconstruction updates, then k-means
on continuous posterior means. Cluster centers initialize regime dynamics and
initial means; predicted cluster transitions initialize the regime network.
Reference labels are not read. ``init_steps=0`` disables this data-only
initialization. Clustering latent positions can be unsuitable for regimes
identified by velocity or another dynamical property instead of position.

Training then uses 1,000 variational updates, batches of eight windows of 64
observations, Adam at ``1e-3``, a four-dimensional latent state and a 16-unit
GRU encoder. These are choices for the single-series setting, not the original
paper's budgets or experimental architecture.

Prediction
----------

The default decoder finds the Viterbi regime path conditional on continuous
posterior means. This is a deterministic plug-in approximation, **not** exact
integration over continuous uncertainty. For the autoregressive posterior,
conditional means are propagated through the forward RNN. Set
``posterior_mean=False`` to use one continuous posterior draw, and
``decoding="marginal"`` to choose pointwise posterior modes.

Stateless noise incorporates the training update and, for autoregressive
sampling, the time index. This avoids parallel-loop random-number ordering
and repeated identical stateless noise at every point. One prediction graph
is reused across calls and sequence lengths. Bitwise equivalence across
devices and TensorFlow versions is not promised.

Representations
---------------

The default uses raw observations. ``feature_window > 1`` replaces each
channel by its centered local mean and standard deviation, with nearest
padding, before normalization. This changes the model's observations: it is
an optional tsseg adaptation for statistical activity regimes, not a
procedure from the paper. Likelihood values of different representations
are not directly comparable.

``downsample`` takes one observation at each specified sampling interval after
feature extraction. Labels are expanded back to the full input length,
including its final points. It trades temporal resolution for a longer
physical context per training window; it is never inferred from labels.

.. code-block:: python

   from tsseg.algorithms import SNLDSDetector

   raw_labels = SNLDSDetector(n_states=4, random_state=0).fit_predict(X)

   # A separate, explicitly selected activity-statistics profile.
   activity_labels = SNLDSDetector(
       n_states=4, feature_window=128, downsample=6, random_state=0
   ).fit_predict(X)

For the paper's posterior family and sampling/decoding policy, use
``inference_type="autoregressive", posterior_mean=False, decoding="marginal"``.
Other differences remain. In particular, windows of one series replace
training on many independent trajectories.

Why a well-fitted model can segment poorly
---------------------------------------------

The regime posterior conditional on a continuous trajectory satisfies

.. math::

   p(s_{1:T}\mid x_{1:T},z_{1:T}) \propto
   p(s_1)p(z_1\mid s_1)
   \prod_{t=2}^{T}p(s_t\mid s_{t-1},x_{t-1})p(z_t\mid z_{t-1},s_t).

The shared emission factors :math:`p(x_t\mid z_t)` cancel from this posterior.
Reconstruction trains the continuous representation, but does not directly
specify the discrete partition. Two regimes can describe rising and falling
parts of each oscillation, while the reference labels describe two long
intervals with different oscillatory statistics. Supplying the reference
number of states cannot resolve this choice of temporal scale.

There is a second failure: a flexible transition can absorb the dynamics into
one state, which then receives nearly all responsibility and gradients.
Dong et al. explicitly discuss this collapse and use annealed regularization
to train and then specialize all transitions. Ansari et al., Appendix B.4.1,
also report collapse and sensitivity to the original objective/regularizer.
Exact marginalization fixes that objective error; it does not guarantee useful
regimes for every series.

Post-validation inspection saved latent trajectories, probabilities and model
weights before and after variational training. The selected failures below
diagnose mechanisms; they are **not** another representative benchmark.
Scores use the model's observation grid, including downsampling for moments.

.. list-table:: Selected failure inspection, seed 0
   :header-rows: 1
   :widths: 27 18 18 18 19

   * - Case
     - Initial state ARI
     - Trained state ARI
     - Initial ELBO/point
     - Trained ELBO/point
   * - TSSB 3, raw
     - -0.0004
     - -0.0001
     - -8.23
     - -0.18
   * - MoCap 1, raw
     - 0.312
     - 0.099
     - -11.23
     - 2.37
   * - MoCap 1, moments
     - 0.599
     - 0.321
     - -29.37
     - 1.78

ELBO estimates use the same 16 training-context windows and four continuous
samples at each stage, with fixed evaluation seeds. They are training-data
diagnostics, not held-out likelihood estimates, and cannot be compared across
representations. In these cases the objective improves while label agreement
does not. In particular:

* **Wrong temporal scale:** TSSB 3 has one annotated change but 383 predicted
  changes. The two states have almost equal occupancy and track the sign of
  the observed derivative (ARI 0.66), rather than the reference (ARI about 0).
  This is not a one-state collapse. Raising the budget to 5,000 updates gives
  ARI 0.001; the autoregressive posterior at 1,000 updates also stays near 0.
  Neither experiment is a full reproduction of the paper's training protocol.
* **Representation versus regime assignment:** on MoCap 1 with moments,
  k-means on the final latent means scores 0.620, while SNLDS regimes score
  0.321. Averaging smoothed probabilities over 16 continuous samples gives
  0.325, and encoding in training-length windows gives 0.322. These controls
  do not recover the latent clustering's label agreement.
* **A lossy decoding approximation:** on raw MoCap 1, averaging the same 16
  sampled posteriors raises ARI from 0.099 at continuous means to 0.256.
  The default plug-in decoder contributes to this failure. It is not the
  explanation for the moments case above or for TSSB 3 (ARI remains about 0
  with the same sampling control).
* **An unsuitable initialization:** on bouncing-ball seed 0, the default
  position-cluster assignments lead to one used state and ARI 0. Removing
  only the assignments to regime dynamics, initial means and transition
  logits gives ARI 0.960. Reconstruction pretraining, its resulting encoder
  outputs and the training-window random sequence are identical. Disabling
  initialization entirely gives 0.965, but also changes those windows and is
  a less controlled comparison. Seed 1 succeeds with the default (ARI 0.922).
  Initialization sensitivity is established here; a universal replacement is
  not established by this selected example.

These results support both optimization failures and a mismatch between
learned dynamical modes and annotated states. They do not establish that the
data cannot be modeled by SNLDS. A minimum-duration constraint, local spectral
representation or different dynamics initialization would impose additional
assumptions and needs its own validation.

The published evaluation also differs materially: it trains across multiple
trajectories; the bouncing-ball appendix uses 100,000 trajectories and 10,000
updates. Its real MoCap experiment trains on 29 salsa trials with 93 observed
coordinates and evaluates a held-out trial qualitatively, without reference
segmentation. That experiment is not evidence of activity-label accuracy on
the four-channel MoCap series bundled here. See the
`paper <https://proceedings.mlr.press/v119/dong20e/dong20e.pdf>`_ and
`experimental appendix <https://proceedings.mlr.press/v119/dong20e/dong20e-supp.pdf>`_.

The reproducible commands, saved summaries and qualifications are in
``examples/diagnostics/README.md`` and ``snlds_mechanism_results.json``.

Runtime
-------

For fixed architectures, training costs :math:`O((I+S)BL)` temporal evaluations,
with initialization updates :math:`I`, variational updates :math:`S`, batch
size :math:`B` and window length :math:`L` (or the series length, if smaller).
Initialization and prediction also encode the full preprocessed series in
:math:`O(n)` temporal evaluations. Discrete forward recursion and Viterbi cost
:math:`O(nK^2)`; the discrete transition MLP costs :math:`O(dK+K^3)` per point.
Local moments use an :math:`O(nd)` uniform filter, without a window matrix.

Defaults process 614,400 window points including initialization, compared
with 4,096,000 for the old defaults. Factorized sampling, narrower transition
networks and removal of unused backward calculations from the training graph
further reduce cost. Timings therefore compare budgets as well as
implementations, not just backends. Prediction reuses its graph.

Fixed validation
----------------

Development used MoCap 0, TSSB 0 and data-seed-0 synthetic mean shifts and a
bouncing ball. Validation uses eight other cases: MoCap 1/2, TSSB 3/6, SKAB 3/6
and the two synthetic tasks with data seed 1. Each series is fitted separately
without labels. The cases were reserved from design choices; they are not
held out from the unsupervised fit on that same series.

Two candidate profiles were fixed before validation: raw defaults, and
``feature_window=128, downsample=6`` on the six real series, both with model
seeds 0/1/2. The pre-audit baseline uses its old defaults with seed 0 on every
case and additionally seeds 1/2 on MoCap 1 and TSSB 3. Only the reference state
count is supplied. This is eight cases and 54 fits, not a full benchmark.

.. include:: ../_includes/snlds_validation.txt

On the six real series, the macro-mean ARI is 0.092 for the old baseline,
0.061 for the new raw defaults and 0.227 for the fixed moment profile. Mean
fit time is 269 s, 46 s and 45 s respectively. BiC is 0.071, 0.252 and 0.502;
SMS is 0.440, 0.436 and 0.578. Thus the raw defaults reduce cost and excessive
boundary switching but do not improve state identification on these real
cases. The moment profile improves every case's mean ARI here, yet TSSB 3
remains near zero and several seeds remain weak. The synthetic bounce has
one collapsed seed. These results do not justify recommending this draft
as a general segmentation-quality improvement.

All seeds, including unsuccessful ones, are retained. Aggregates do not
choose the best profile or seed per case. Single-run baseline cells cannot
estimate seed variance. Four jobs ran concurrently, with one thread per
process; these are diagnostic wall times, not isolated hardware benchmarks.
The numerical implementation was frozen during evaluation. An interrupted
first run lost its temporary artifacts, so only the complete persistent rerun
is used in the table and raw validation record.

``examples/diagnostics/snlds_validate.py`` reruns the protocol. Its companion
README explains preparing the external arrays and comparing checkouts.
``snlds_validation_results.json`` records actual parameters, input and code
hashes, and every run. The bouncing ball is inspired by the published task,
not a reproduction of its training corpus.

Numerical audit and history
---------------------------

The audit reproduced these defects before correcting them:

* The pairwise posterior uses destination/source axes, but the complete-data
  score attached emissions to source states. Enumerating eight paths of a
  two-state, three-point model gives -2.692044; the old code gave -7.117615.
* Even with the axis fixed, omitting discrete posterior entropy gave
  -2.692044 instead of the marginal log likelihood -1.907844. Both objective
  values and gradients now match independent path enumeration.
* Covariances were transformed outside the gradient tape and were not
  Keras-tracked. Scalar softmax also forced variances to be constant even
  when trainable. Softplus parameters now have live gradients in one dimension.
* Initial Gaussian means and discrete logits were not optimized on Keras 3.
  They are now tracked; initial probabilities are normalized on every use.
* The LSTM compatibility fallback supplied only one state instead of hidden
  and cell states. The cell now creates its own state structure.
* Stateful sampling inside parallel loops made runs differ despite a seed.
  Latent draws now use explicit stateless per-update and per-time seeds.

A historical 54-series pilot scored below random segmentation (BiC 0.074
against 0.377, ARI 0.066 against 0.208), with about 200 times too many changes.
It predates the numerical fixes. The first correction-only candidate
(``f0d9654``) also deteriorated in two single-seed diagnostics: mean-shift ARI
0.813 to 0.095 and MoCap 0 ARI 0.118 to 0.033. Those raw measurements remain in
``snlds_audit_results.json`` and do not evaluate the current architecture.

Limitations
-----------

The likelihood objective does not generally identify semantic activity
labels. One continuous model can explain multiple activities with one regime,
or an activity can split into primitive dynamical phases. Initialization,
latent dimension and temporal context matter. A requested state count does
not guarantee that all states will be used after training.

There is no explicit duration model or minimum dwell time. The original
MoCap experiment trained on 29 salsa trajectories with 93 coordinates and
qualitatively assessed a held-out trajectory without reference segmentation.
This differs from fitting one four-channel series and evaluating semantic
activity labels. Neither numerical fixes nor this pilot establish suitability
for every such dataset. Explicit hyperparameter configurations in external
benchmark repositories also need reevaluation; constructor defaults do not
override values supplied by the caller.

**Source:** modified Google Research TensorFlow code (Apache-2.0), with
Gaussian parameters, factorized inference, decoding and initialization
implemented in tsseg. The reference is
`Dong et al., ICML 2020 <https://proceedings.mlr.press/v119/dong20e.html>`_
and its `supplement <https://proceedings.mlr.press/v119/dong20e/dong20e-supp.pdf>`_.

API reference
-------------

.. automodule:: tsseg.algorithms.snlds.detector
   :members:
   :show-inheritance:
   :undoc-members:
