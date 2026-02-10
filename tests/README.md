# Tests

This folder contains the test suite for the `tsseg` project. Tests are
written using **pytest** and are organized into two areas:

- **Root-level tests** — validate public modules (datasets, metrics).
- **`tests/algorithms/`** — generic tests for every detector registered in
  `tsseg.algorithms.__all__`, driven by shared fixtures.

## Structure

```
tests/
├── algorithms/
│   ├── conftest.py            # Shared fixtures & ALGORITHM_CONFIGS registry
│   ├── test_fit_predict.py    # Output existence, shape, detection quality
│   ├── test_instantiation.py  # Default & override construction
│   ├── test_tags.py           # Required tag presence
│   └── __init__.py
├── test_datasets.py           # MoCap loader (shapes, errors, trial selection)
└── test_metrics.py            # All metrics (F1, Covering, ARI, SMS, …)
```

## How it works

### Algorithm tests (`tests/algorithms/`)

`conftest.py` provides:

- **`ALGORITHM_CONFIGS`** — a dict mapping each algorithm name to an
  `AlgorithmConfig` that declares optional dependencies, init kwargs,
  semi-supervised mode, etc.  Algorithms not listed fall back to
  `DEFAULT_CONFIG` (no deps, default init).
- **`synthetic_series`** — a session-scoped fixture generating univariate
  and multivariate segmented signals with known change points.
- **`fit_predict_result`** — parametrised over `__all__`; instantiates
  each algorithm, runs `fit` / `predict`, and skips gracefully when an
  optional dependency is missing.

Each test function (`test_*_output_exists`, `test_*_output_shape`,
`test_*_detects_change_points`) runs for every algorithm × data shape
combination automatically.

### Adding a new algorithm to the test suite

1. Register the detector class in `tsseg.algorithms.__all__`.
2. If it needs special init args, optional deps, or semi-supervised labels,
   add an entry in `ALGORITHM_CONFIGS` inside `conftest.py`.
3. Run `pytest tests/algorithms/` — the generic tests cover it.

## Running

```bash
pytest                              # full suite
pytest tests/test_metrics.py        # metrics only
pytest tests/algorithms/ -k Pelt    # one algorithm
pytest -x --tb=short                # stop at first failure
```

Pytest automatically **skips** tests whose optional dependencies are
missing.
