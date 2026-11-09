SNLDS
=====

Switching non-linear dynamical system trained with collapsed amortized
variational inference (CAVI-SNLDS).

.. warning::
   **Experimental.** In a historical pilot on 54 series of TSB-SEG (TSSB,
   UTSA, SKAB, MoCap), SNLDS scored below a random segmentation
   (bidirectional covering 0.074 against 0.377, ARI 0.066 against 0.208): its
   labels change about 200 times more often than the true regimes. Training five
   times longer did not help on the two series tried. It is kept for further
   study, not as a recommended detector. These measurements predate the
   numerical corrections described below and do not evaluate the corrected
   implementation.

Description
-----------

SNLDS models a time series with a discrete regime :math:`s_t \in \{1, \dots,
K\}`, a continuous latent state :math:`z_t` and an observation :math:`x_t`.
Each regime has its own non-linear transition :math:`p(z_t \mid z_{t-1}, s_t)`,
an MLP; the regime switches follow :math:`p(s_t \mid s_{t-1}, x_{t-1})`, and
:math:`x_t` is emitted from :math:`z_t` by a shared network.

Training maximises an evidence lower bound. An RNN infers the continuous states
:math:`q(z_{1:T} \mid x_{1:T})`, and the discrete states are integrated out
exactly with the forward-backward algorithm.

After training, each time step receives the regime of highest posterior
probability :math:`p(s_t \mid x_{1:T}, z_{1:T})`.

Two devices from the paper prevent the collapse onto a single regime: a
cross-entropy regulariser between the regime posterior and a uniform prior,
with weight :math:`\beta`, and a temperature :math:`\tau` on the discrete
transitions. Both start large and decay, :math:`\beta` first, then
:math:`\tau` down to 1. tsseg follows the schedule of the paper's Reacher
experiment (:math:`\beta_0 = \tau_0 = 1000`, decay by 0.975 every 500 steps
after steps 50,000 and 100,000, out of 300,000), scaled to ``n_train_steps``:
the decays start after ``n_train_steps / 6`` and ``n_train_steps / 3`` steps,
with a factor 0.975 every ``n_train_steps / 600`` steps.

Each gradient step uses ``batch_size`` windows of ``train_window`` points drawn
at random positions of the series, which bounds the cost of a step; prediction
runs on the whole series. The paper trains on minibatches of 32 short
sequences (50 to 100 steps) from many trajectories; tsseg draws its windows from
the single series it segments.

| **Type:** state detection
| **Supervision:** semi-supervised (``n_states`` is the number of regimes)
| **Scope:** univariate and multivariate
| **Complexity:** :math:`O(S\,B\,L)` time steps for training, independent of
  *n* (*S* is ``n_train_steps``, *B* ``batch_size``, *L* ``train_window``, or
  *n* if smaller), then :math:`O(n)` time steps for the prediction pass. Each
  time step evaluates the *K* continuous transition networks (*K* is
  ``n_states``), the discrete transition network, :math:`O(d\,K^{2} + K^{4})`,
  and the forward-backward recursion, :math:`O(K^{2})`
| **Requires:** TensorFlow, TensorFlow Probability and ``tf-keras``
  (``pip install tsseg[snlds]``)

.. note::

   Training is slow. A step processes :math:`B\,L` time steps for ``batch_size``
   :math:`B` windows of :math:`L` points, independent of the length of the
   series: about
   0.1 s on one CPU thread with the defaults, so about 3 minutes for the 2,000
   default steps. The first call also spends a few seconds compiling the training
   graph.

.. note::

   Training is not reproducible from run to run. Two fits with the same
   ``random_state`` start from the same weights and draw the same windows, but
   TensorFlow does not compute the training identically in every run, even after
   ``tf.config.experimental.enable_op_determinism()``, and over 2,000 steps the
   differences can lead to another model. On a series of three segments with
   clear mean shifts (1,000 points), historical runs before the numerical fixes gave
   adjusted Rand indices of 0.54, 0.97 and 0.54, and seeds 0 to 4 gave 0.50 to
   0.97, two of them above 0.8: the other runs merged two regimes. Prediction
   samples the continuous states once, so two calls to ``predict`` on one fitted
   model can still differ on a few labels.

Implementation audit
--------------------

The November 2026 audit found four reproducible implementation errors:

* The pairwise posterior uses axes ``[destination, source]``, but the expected
  complete-data likelihood broadcast the current emission over the source
  state. A two-state, three-point example with eight enumerated paths gives
  an expected complete-data log likelihood of -2.692044; the old code gave
  -7.117615.
* The objectives called ``elbo`` and ``iwae`` used the expected complete-data
  score instead of the log likelihood with discrete paths marginalized out.
  This omits the entropy of the discrete posterior. Differentiating through
  that posterior also changes the gradient. Even after fixing the state axis,
  the example above gave -2.692044 instead of the exact marginal log likelihood
  -1.907844. Both objectives now use the sum of the forward normalizers, with
  the continuous posterior entropy or importance weights as appropriate.
  ``initial_likelihood`` and ``sequence_likelihood`` remain diagnostics of the
  expected complete-data score.
* Gaussian scales were transformed into eager tensors in the constructors,
  outside the training gradient tape. Their raw parameters were also absent
  from Keras 3's tracked weights. ``trainable_cov=True`` therefore did not
  train the continuous transition covariance. Raw scales are now Keras
  weights, transformed when used. Covariances configured as fixed stay fixed.
* The Keras 3 fallback for ``lstm`` created one initial state instead of the
  hidden and cell states, so fitting failed. The cell now supplies its own
  initial state structure.

These errors originate in the vendored calculations or their Keras port.
They prevent attributing the historical pilot's poor scores entirely to the
method or the data. The regression tests check values and gradients against
enumerated paths, covariance updates under ``tf.function``, and all three RNN
types. They do not establish segmentation quality on a benchmark.

Small before/after diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With 2,000 updates, seed 42, the true state count and otherwise default
parameters, a diagnostic on Python 3.12 / TensorFlow 2.21 / Keras 3.14 gave:

.. list-table::
   :header-rows: 1

   * - Series
     - Implementation
     - ARI
     - BiC
     - Predicted changes
     - Fit (s)
   * - Mean shifts (1,000 points, 2 true changes)
     - Before
     - 0.813
     - 0.710
     - 99
     - 239
   * - Mean shifts
     - Corrected
     - 0.095
     - 0.009
     - 653
     - 242
   * - MoCap trial 0 (4,579 points, 6 true changes)
     - Before
     - 0.118
     - 0.090
     - 506
     - 267
   * - MoCap trial 0
     - Corrected
     - 0.033
     - 0.037
     - 2,141
     - 269

Numerical correctness did **not** improve these two segmentations: both
accuracy and oversegmentation became worse in these runs. Prediction took
about one second. Two predictions of each fitted model disagreed on at most
0.4% of labels, much less than the amount of oversegmentation. The fit used
one CPU thread per process; runs were concurrent, so these times are not an
isolated performance benchmark. There is one run per case and implementation,
and training is nondeterministic: this is not an estimate of mean performance
or an ablation separating the individual corrections.

The correction is a candidate for further evaluation, not evidence that this
detector is now a useful baseline. In particular, increasing training time or
changing defaults is not justified by this diagnostic alone.

``examples/diagnostics/snlds_audit.py`` regenerates these measurements;
``snlds_audit_results.json`` beside it records the raw results and source hashes.
Run the same script on the baseline commit (``e0b09b3``) and on the corrected
checkout, for example:

.. code-block:: console

   python -m examples.diagnostics.snlds_audit --case mean-shifts
   python -m examples.diagnostics.snlds_audit --case mocap

Why training is slow
~~~~~~~~~~~~~~~~~~~~

The defaults process :math:`2000 \times 8 \times 256 = 4\,096\,000` latent time
steps for each series. Each window requires an autoregressive RNN with
sampling, forward and backward recursions over discrete states, and reverse
mode differentiation. These time recursions limit parallel execution even
for small networks. The discrete transition MLP has a hidden layer of
:math:`4K^2` units and :math:`K^2` outputs, hence :math:`4K^4` weights in that
layer alone: supplying many states can be expensive. Compilation is an
additional cost for each newly fitted detector; prediction also creates a
new compiled function on each call.

Windows already bound training cost independently of series length. More
training steps increase cost linearly and do not repair a wrong objective.
The corrections above address correctness; no general speedup is promised.

Method and data limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The wrapper is an adaptation of the authors' example, not a reproduction of
the architectures and training regimes in their experiments. In particular,
``network_input_embedding`` is the identity: the continuous posterior sees
past observations through a forward RNN, without the paper's bidirectional
encoder. The discrete posterior still uses forward-backward smoothing.
The default inference RNN has four units, and training compresses the Reacher
annealing schedule into 2,000 updates at a fixed learning rate.

The paper's MoCap experiment trained on 29 salsa trajectories, using 93
coordinates per frame, and evaluated a held-out trajectory qualitatively,
without ground-truth segmentation. Its states describe primitive directions
of motion. Fitting windows from a single four-channel MoCap series and
scoring activity labels is a different experiment. This makes data diversity
and the meaning of the target states plausible limitations; it does not
prove that switching dynamics cannot describe the benchmark signals.

There is no explicit segment-duration prior or minimum dwell time. Pointwise
posterior argmax on one continuous-state sample can produce rapid switching;
learning good reconstruction or dynamics does not guarantee matching semantic
activity labels. The default emission standard deviation is fixed at 0.05
for standardized univariate input, and the latent dimension is eight even
for one observed channel. The suitability of these assumptions needs an
ablation, rather than increasing training time alone.

Known residual issues include stateful sampling in a parallel loop (run-to-run
variation), and initial-state prior parameters that are not tracked as Keras
3 weights. This correction does not validate the entire model as a faithful
reproduction of the paper. Defaults, decoding policy and estimator tags are
unchanged.

The mathematical reference is
`Dong et al., ICML 2020 <https://proceedings.mlr.press/v119/dong20e.html>`_,
especially Eq. (12) and the experimental details in the
`supplement <https://proceedings.mlr.press/v119/dong20e/dong20e-supp.pdf>`_.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 55

   * - Name
     - Type
     - Default
     - Description
   * - ``n_states``
     - int
     - ``3``
     - Number of discrete regimes :math:`K`.
   * - ``hidden_dim``
     - int
     - ``8``
     - Dimension of the continuous latent state :math:`z_t`.
   * - ``rnn_dim``
     - int
     - ``4``
     - Hidden size of the inference RNN.
   * - ``rnn_type``
     - str
     - ``"simplernn"``
     - Inference RNN cell (``"gru"``, ``"lstm"``, ``"simplernn"``).
   * - ``n_train_steps``
     - int
     - ``2000``
     - Number of gradient steps; the schedules scale with it.
   * - ``learning_rate``
     - float
     - ``1e-4``
     - Adam learning rate.
   * - ``batch_size``
     - int
     - ``8``
     - Training windows per gradient step.
   * - ``train_window``
     - int / None
     - ``256``
     - Length of the training windows (``None``: the whole series).
   * - ``objective``
     - str
     - ``"elbo"``
     - Training objective (``"elbo"`` or ``"iwae"``).
   * - ``use_temperature_annealing``
     - bool
     - ``True``
     - Anneal the temperature of the discrete transitions down to 1.
   * - ``temperature_init``
     - float
     - ``1000.0``
     - Initial temperature.
   * - ``use_cross_entropy_reg``
     - bool
     - ``True``
     - Add the cross-entropy regulariser on regime usage.
   * - ``xent_init``
     - float
     - ``1000.0``
     - Initial weight of the regulariser.
   * - ``random_state``
     - int / None
     - ``42``
     - Seed of the training windows, of the initial weights and of TensorFlow
       (training still differs from run to run, see the note above).
   * - ``verbose``
     - int
     - ``0``
     - Log the objective every tenth of training when ``>= 1``.
   * - ``axis``
     - int
     - ``0``
     - Time axis.

Usage
-----

.. code-block:: python

   from tsseg.algorithms import SNLDSDetector

   detector = SNLDSDetector(n_states=3)
   labels = detector.fit_predict(X)

**Implementation:** wrapper around the authors' TensorFlow code (Google
Research, Apache License 2.0), vendored in
``tsseg/algorithms/snlds/vendor`` with relative imports, Keras 3 compatibility
fixes, corrected collapsed objectives and trainable Gaussian scales.

**Reference:** Dong, Seybold, Murphy & Bui (2020), *Collapsed Amortized
Variational Inference for Switching Nonlinear Dynamical Systems*, ICML.

API reference
-------------

.. automodule:: tsseg.algorithms.snlds.detector
   :members:
   :show-inheritance:
   :undoc-members:
