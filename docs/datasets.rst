Datasets
========

``tsseg`` ships one small dataset, used in the examples and tests, and works
with any series given as ``(X, y)`` arrays. The benchmark datasets of the
paper (TSSB, UTSA, SKAB, ...) are not bundled.

Built-in dataset
----------------

``load_mocap`` loads one of the 9 trials of subject 86 of the
`CMU Graphics Lab Motion Capture Database <https://mocap.cs.cmu.edu/>`_
(4 channels: left and right arms and legs), with one activity label per time
point. It returns a tuple ``(X, y)``:

* ``X`` — a 2-D array of shape ``(n_timepoints, n_channels)``
* ``y`` — an integer array of state labels aligned with ``X``

Example:

.. code-block:: python

   from tsseg.data.datasets import load_mocap

   X, y = load_mocap(trial=0)  # an index in 0..8, or a trial ID such as "07"
   print(X.shape, y.shape)  # (4579, 4) (4579,)

The ground truth is given as state labels. The change points are the indices
where the label changes, given by :func:`tsseg.algorithms.utils.extract_cps`
(without ``0`` and the series length, like the output of the change point
detectors):

.. code-block:: python

   from tsseg.algorithms.utils import extract_cps

   change_points = extract_cps(y)

Your own data
-------------

Detectors take a NumPy array of shape ``(n_timepoints,)`` or
``(n_timepoints, n_channels)``, a ``pandas.Series`` or a
``pandas.DataFrame`` (one column per channel). If your annotations are
change points, turn them into state labels (one segment, one label) to
evaluate state detectors:

.. code-block:: python

   import numpy as np

   n_timepoints, change_points = 1000, [300, 700]
   y = np.zeros(n_timepoints, dtype=int)
   for label, start in enumerate(change_points, start=1):
       y[start:] = label

For large datasets, prefer storing them in an external location and provide a
lazy loader to avoid shipping them inside the package.

API reference
-------------

.. automodule:: tsseg.data.datasets
   :members:
   :show-inheritance:
   :undoc-members:
