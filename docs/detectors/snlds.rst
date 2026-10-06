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

.. warning::

   The paper prevents the collapse onto a single regime with two annealed
   devices: a cross-entropy regulariser between the regime posterior and a
   uniform prior, with weight :math:`\beta`, and a temperature :math:`\tau` on
   the discrete transitions. Both start large and decay to 0 and 1; in the
   paper's Reacher experiment they start at 1,000 and decay after 50,000 and
   100,000 steps respectively, over 300,000 training steps. This wrapper takes
   the schedule values of the upstream Lorenz example instead, whose initial
   values are 0. As a result :math:`\tau` stays at 1 and :math:`\beta` at
   :math:`10^{-10}` throughout training, even with
   ``use_temperature_annealing=True`` and ``use_cross_entropy_reg=True``: neither
   device is active.

| **Type:** state detection
| **Supervision:** semi-supervised (``n_states`` is the number of regimes)
| **Scope:** univariate and multivariate
| **Requires:** TensorFlow, TensorFlow Probability and ``tf-keras``
  (``pip install tsseg[snlds]``)

.. note::

   Training is slow. Each gradient step runs a forward-backward pass over the
   whole series, so a step costs :math:`O(nK^2)`. With the default 5,000 steps,
   none of the nine MoCap series (4,600 to 10,600 points) finished within
   10 minutes on one CPU thread. The first call also spends about 15 s compiling
   the training graph.

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
     - ``5000``
     - Number of gradient steps.
   * - ``learning_rate``
     - float
     - ``1e-4``
     - Adam learning rate.
   * - ``batch_size``
     - int
     - ``1``
     - Sequences per batch (one series is segmented at a time).
   * - ``objective``
     - str
     - ``"elbo"``
     - Training objective (``"elbo"`` or ``"iwae"``).
   * - ``use_temperature_annealing``
     - bool
     - ``True``
     - Anneal the temperature of the discrete transitions.
   * - ``use_cross_entropy_reg``
     - bool
     - ``True``
     - Add the cross-entropy regulariser on regime usage.
   * - ``random_state``
     - int / None
     - ``42``
     - Seed for TensorFlow and NumPy.
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

   detector = SNLDSDetector(n_states=3, n_train_steps=2000)
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
