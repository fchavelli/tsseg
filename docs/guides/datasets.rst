.. _guides-datasets:

Datasets
========

``tsseg`` packages helpers to access curated datasets used in the test suite
and benchmarks. You can browse available datasets via
:mod:`tsseg.datasets`.

.. code-block:: python

   from tsseg import datasets

   names = datasets.list_datasets()
   print(names)
   X, y = datasets.load_dataset("synthetic_regimes")

Each loader returns a tuple ``(X, y)``:

* ``X`` – a 2D array with shape ``(n_timepoints, n_channels)``.
* ``y`` – an array of integer state labels aligned with ``X``.

When a dataset ships with annotated change points, you can convert labels to
change-point indices using :func:`tsseg.algorithms.utils.extract_cps`.

### Creating custom datasets

You can wrap your own time series as lightweight dataset objects by exposing a
callable returning ``(X, y, metadata)``. For testing you may reuse the synthetic
fixtures in ``tests/algorithms/conftest.py``:

.. code-block:: python

   from tsseg.tests.algorithms.conftest import synthetic_series

   series = synthetic_series()
   X = series["multivariate"]["X"]
   y = series["multivariate"]["y"]

For large datasets, prefer storing them in an external location and provide a
lazy loader to avoid shipping them inside the package.
