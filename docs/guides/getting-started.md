# Getting started

This quick guide shows how to install `tsseg`, load a time series and run a
segmentation detector. The instructions assume Python 3.10+ and a virtual
environment.

## Installation

This library relies on a robust environment. It is recommended to use `conda` or `mamba`.

```bash
# Clone the repository
git clone https://github.com/fchavelli/tsseg.git
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

Change point detection with `PeltDetector`, which needs no optional
dependency and does not need to know the number of change points:

```python
import numpy as np
from tsseg.algorithms import PeltDetector

# Toy signal: three segments with different means
rng = np.random.default_rng(42)
seg_lengths = [200, 150, 250]
segments = [
    rng.normal(mu, 0.2, length) for mu, length in zip([-0.5, 0.8, 0.0], seg_lengths)
]
series = np.concatenate(segments)[:, None]  # shape (n_points, n_channels)

change_points = PeltDetector(pen_scale="bic").fit_predict(series)
print(change_points)  # first index of each new segment: [200 350]
```

State detection returns one label per time point instead. With
`GaussianHMMDetector`, the number of states is chosen by BIC:

```python
from tsseg.algorithms import GaussianHMMDetector

states = GaussianHMMDetector(random_state=0).fit_predict(series)
print(np.unique(states, return_counts=True))
```

Most detectors expose the `fit`, `predict`, and `fit_predict` methods and share
metadata tags (see [Detectors](../detectors/index)).

## Guided (semi-supervised) usage

When the number of change points or states is known, detectors with the
`capability:semi_supervised` tag take it in their constructor (the parameter
name depends on the detector: `n_cps`, `n_segments`, `n_states`, ...):

```python
from tsseg.algorithms import DynpDetector

print(DynpDetector.get_class_tag("capability:semi_supervised"))  # True
change_points = DynpDetector(n_cps=2).fit_predict(series)
states = GaussianHMMDetector(n_states=3, random_state=0).fit_predict(series)
```

**Important:** `y` is not used to infer these parameters. Do not pass `y` to
`fit` if you intend to run in unsupervised mode; set parameters like `n_cps`,
`n_segments` or `n_states` in the constructor.

## Running the test suite

To ensure your environment is healthy (the full suite takes a long time; select
tests with `-k`):

```bash
python -m pytest tests/
python -m pytest tests/algorithms -k PeltDetector
```

`make test`, `make lint` and `make docs` run the same commands as the test,
lint and documentation jobs of the CI (select tests with
`make test PYTEST_ARGS="-k PeltDetector"`).
