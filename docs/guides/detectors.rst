.. _guides-detectors:

Detectors overview
==================

The :mod:`tsseg.algorithms` package exposes a wide range of segmentation
algorithms with a shared interface derived from
:class:`aeon.segmentation.base.BaseSegmenter`.

All detectors share the ``fit``/``predict``/``fit_predict`` API and expose
metadata through the :py:meth:`~aeon.base.BaseEstimator.get_tag` method. Common
tags include:

* ``capability:univariate`` – accepts single-channel time series.
* ``capability:multivariate`` – handles multi-channel inputs.
* ``returns_dense`` – ``True`` if predictions are sparse change points,
  ``False`` when they are dense state labels.
* ``semi_supervised`` – ``True`` when the algorithm can leverage labels during fit.

.. note::
   In ``tsseg``, the ``fit`` method signature is ``fit(X, y=None)``.
   However, for **unsupervised** learning, ``y`` should strictly be ``None``.
   Passing ``y`` is reserved for **semi-supervised** or **supervised** scenarios
   where the algorithm explicitly supports it.
   
   Previously, some algorithms inferred parameters (like the number of segments)
   from ``y`` inside ``fit``. This behavior has been deprecated.
   You must now explicitly pass such parameters (e.g., ``n_segments``, ``n_states``)
   to the detector's constructor.

Example usage:

.. code-block:: python

   from tsseg.algorithms import EspressoDetector
   from tsseg.metrics import F1Score

   # Explicitly provide n_segments if known, or use a default/heuristic
   detector = EspressoDetector(subsequence_length=32, chain_len=5, n_segments=3)
   
   # Unsupervised fit (y is not passed)
   y_pred = detector.fit_predict(X)

   f1 = F1Score()
   result = f1.compute(y_true, y_pred)
   print(f"F1 score: {result['f1']:.3f}")

Refer to the :ref:`api-index` for detailed class reference. Recently, the
classic ``ruptures`` change point detectors (``BinSegDetector``,
``BottomUpDetector``, ``DynpDetector``, ``KernelCPDDetector``, ``PeltDetector``
and ``WindowDetector``) were vendored into ``tsseg.algorithms.ruptures`` so
they are always available without installing an extra dependency. The new
``TSCP2Detector`` mirrors the upstream TensorFlow implementation and requires
the optional ``tscp2`` extra (installs ``tensorflow`` and ``tcn``). ``HdpHsmmDetector`` now implements a
pure NumPy/SciPy sticky HDP-style segmenter and no longer requires
the ``pyhsmm`` dependency. The legacy ``pyhsmm``-backed detectors are
deprecated and will be removed in a future release.
When adding new algorithms, ensure they set the appropriate tags and register
themselves in ``tsseg.algorithms.__all__`` so the documentation and tests can
discover them.

.. _testing-contract:

Estimator checks
^^^^^^^^^^^^^^^^

Every algorithm in ``__all__`` is automatically validated by the test suite
in ``tests/algorithms/``.  The checks are inspired by
``sklearn.utils.estimator_checks.check_estimator`` and ensure that all
detectors satisfy the ``BaseSegmenter`` API contract.

The test suite verifies **four categories**:

1. **Common contract** (``test_common.py``): ``get_params`` / ``set_params``
   round-trip, ``clone``, ``fit`` returns ``self``, ``is_fitted`` lifecycle,
   ``NotFittedError``, ``reset``, ``repr``, pickle round-trip.

2. **Tag contract** (``test_tags.py``): required tags are present and
   consistent (``returns_dense`` ↔ ``detector_type``).

3. **Prediction contract** (``test_predict.py``): output shape, dtype, value
   range, determinism, ``fit_predict`` ≡ ``fit`` + ``predict``.

4. **Input validation** (``test_input_validation.py``): invalid types are
   rejected, capability mismatches raise ``ValueError``.

If your algorithm needs custom constructor arguments or has optional
dependencies, declare an ``AlgorithmOverride`` in
``tests/algorithms/conftest.py``:

.. code-block:: python

    from tests.algorithms.conftest import AlgorithmOverride

    ALGORITHM_OVERRIDES["MyNewDetector"] = AlgorithmOverride(
        init_kwargs={"n_segments": 5},
        dependencies=("torch",),
    )

Run the full suite with:

.. code-block:: bash

    pytest tests/algorithms/ -v
