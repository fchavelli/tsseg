# Getting started

This quick guide shows how to install `tsseg`, load a time series and run a
segmentation detector. The instructions assume Python 3.9+ and a virtual
environment.

## Installation

This library relies on a robust environment. It is recommended to use `conda` or `mamba`.

```bash
# Clone the repository
git clone https://github.com/fchavell/tsseg.git
cd tsseg

# Install the environment using the provided makefile
make install
conda activate tsseg-env
```

If you prefer `pip` and want to install it as a library in an existing environment:

```bash
pip install -e .[all]
```

## Basic usage

This example demonstrates how to use `AutoPlaitDetector`, an unsupervised algorithm for regime discovery.

```python
import numpy as np
from tsseg.algorithms import AutoPlaitDetector

# Generate a toy signal
rng = np.random.default_rng(42)
seg_lengths = [200, 150, 250]
segments = [rng.normal(mu, 0.2, length) for mu, length in zip([-0.5, 0.8, 0.0], seg_lengths)]
series = np.concatenate(segments)[:, None]  # shape (n_points, n_channels)

# Run the detector
# Note: We initialize the detector with any necessary parameters.
# Unsupervised algorithms do not use 'y' in fit().
model = AutoPlaitDetector()
state_labels = model.fit_predict(series)
print(np.unique(state_labels, return_counts=True))
```

Most detectors expose the `fit`, `predict`, and `fit_predict` methods and share
metadata tags (see :doc:`guides/detectors`).

## Semi-supervised usage

If an algorithm supports semi-supervised learning (check its `semi_supervised` tag),
you can pass labels to `fit`:

```python
# Example for a semi-supervised detector
# model = SomeSemiSupervisedDetector(n_states=3)
# model.fit(X, y=labels)
# preds = model.predict(X)
```

**Important:** Do not pass `y` to `fit` if you intend to run in unsupervised mode.
Previous versions might have inferred parameters from `y`; this is no longer supported.
Explicitly set parameters like `n_segments` or `n_states` in the constructor.

## Running the test suite

To ensure your environment is healthy:

```bash
make test
```

Or directly:

```bash
pytest tests/
```
