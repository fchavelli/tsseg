SNLDS
=====

Switching non-linear dynamical system trained with collapsed amortized
variational inference (CAVI-SNLDS).

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
| **Requires:** TensorFlow, TensorFlow Probability and ``tf-keras``
  (``pip install tsseg[snlds]``)

.. note::

   Training is slow. A step costs :math:`O(BLK^2)` for ``batch_size`` :math:`B`
   windows of :math:`L` points, independent of the length of the series: about
   0.1 s on one CPU thread with the defaults, so about 3 minutes for the 2,000
   default steps. The first call also spends a few seconds compiling the training
   graph.

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
     - Seed of the training windows and of TensorFlow (a few labels can still
       differ between calls).
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
``tsseg/algorithms/snlds/vendor`` with relative imports and a Keras 3 fix of the
inference RNN.

**Reference:** Dong, Seybold, Murphy & Bui (2020), *Collapsed Amortized
Variational Inference for Switching Nonlinear Dynamical Systems*, ICML.

API reference
-------------

.. automodule:: tsseg.algorithms.snlds.detector
   :members:
   :show-inheritance:
   :undoc-members:
