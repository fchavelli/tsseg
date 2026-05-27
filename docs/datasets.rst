Datasets
========

``tsseg`` ships a small collection of curated datasets used in the test suite
and benchmarks, plus helpers to load your own data through the unified
``(X, y)`` interface.

Built-in datasets
-----------------

Each loader returns a tuple ``(X, y)``:

* ``X`` — a 2-D array of shape ``(n_timepoints, n_channels)``
* ``y`` — an integer array of state labels aligned with ``X``

Example:

.. code-block:: python

   from tsseg.data.datasets import load_mocap

   X, y = load_mocap(trial=0)
   print(X.shape, y.shape)

When a dataset ships with annotated change points, you can convert dense
labels into change-point indices via
:func:`tsseg.algorithms.utils.extract_cps`.

Custom datasets
---------------

You can wrap your own time series as lightweight dataset objects by exposing a
callable returning ``(X, y, metadata)``. For testing, reuse the synthetic
fixtures in ``tests/algorithms/conftest.py``:

.. code-block:: python

   from tests.algorithms.conftest import synthetic_series

   series = synthetic_series()
   X = series["multivariate"]["X"]
   y = series["multivariate"]["y"]

For large datasets, prefer storing them in an external location and provide a
lazy loader to avoid shipping them inside the package.

API reference
-------------

.. automodule:: tsseg.data.datasets
   :members:
   :show-inheritance:
   :undoc-members:
