# Tests

The test suite of `tsseg`, written with **pytest**. It has three parts:

- **`tests/algorithms/`**: a generic suite that runs on every detector listed
  in `tsseg.algorithms.__all__`, plus detector-specific tests.
- **`tests/reproduction/`**: published results reproduced on datasets
  downloaded on demand (see its `README.md`); deselected by default.
- **Root-level tests**: the other public modules.

## Structure

```
tests/
├── algorithms/
│   ├── conftest.py                     # detector discovery, overrides, guidance, synthetic data
│   ├── test_common.py                  # estimator contract (get/set_params, clone, fit, reset, pickle)
│   ├── test_predict.py                 # output contract, fit_predict consistency, determinism
│   ├── test_tags.py                    # tags
│   ├── test_input_validation.py        # invalid inputs, pandas inputs
│   ├── test_docstrings.py              # the class docstring documents the constructor
│   ├── test_guided_no_change_point.py  # guided mode, including on a single regime
│   ├── test_quality.py                 # detection quality on the synthetic series (slow)
│   └── test_<detector>.py              # tests of one detector
├── reproduction/                       # published results on downloaded data
└── test_*.py                           # metrics, parameter schemas, base class, utilities,
                                        # ruptures wrappers and backend, MoCap loader
```

## The generic detector suite

`tests/algorithms/conftest.py` provides:

- **`algorithm`**: a fixture parametrised over `tsseg.algorithms.__all__`,
  which yields `(name, cls, override, instance)`. A detector is skipped when
  one of its optional dependencies is missing or when it cannot be imported.
- **`ALGORITHM_OVERRIDES`**: an `AlgorithmOverride` per detector that a bare
  `Cls()` does not suit: constructor arguments (`init_kwargs`), optional
  dependencies to skip on (`dependencies`, which must match the
  `python_dependencies` tag, see `test_tags.py`), guided mode
  (`semi_supervised`) or an unconditional skip (`skip_reason`). Detectors
  without an entry are built with their default constructor.
- **`apply_supervision(instance, y)`**: guided mode. It sets the constructor
  parameters listed in `SUPERVISION_PARAMS` (`n_cps`, `n_segments`,
  `n_states`, `k_max`, `max_cps`, …) to the number of change points, segments
  or states of the labels `y`. The generic tests call it, then `fit(X, y)`,
  for the detectors whose override has `semi_supervised=True` (those without
  an unsupervised mode).
- **`synthetic_data`**: a session-scoped fixture with a univariate and a
  2-channel series of 1000 points, three Gaussian segments (change points at
  300 and 700) and their state labels.

Every test function that takes the `algorithm` fixture runs once per
detector.

`test_quality.py` asks each detector, in guided mode when it has one, to find
the segments of the `synthetic_data` series: F1 score of at least 0.75 with
a margin of 50 points, or adjusted Rand index of at least 0.8. The detectors
below this bar are listed in its `KNOWN_FAILURES`, with what they returned;
their test is an `xfail`, strict for deterministic detectors, so a fix shows
up as a failing test until its entry is removed.

### Adding a detector to the suite

1. Export the detector in `tsseg.algorithms.__all__`: the generic suite then
   tests it.
2. If its default constructor does not suit the tests (no stopping
   criterion, slow defaults, optional dependencies), add an entry to
   `ALGORITHM_OVERRIDES`.
3. If it has a guided mode (`capability:semi_supervised`) and takes the count
   under a name absent from `SUPERVISION_PARAMS`, add the name there;
   `test_supervision_sets_a_constructor_parameter` fails otherwise.
4. Put its own tests in `tests/algorithms/test_<detector>.py` and run
   `pytest tests/algorithms -k <DetectorClass>`.

## Running

```bash
make test                           # same command as the CI test job
pytest                              # full suite
pytest tests/test_metrics.py        # one module
pytest tests/algorithms -k Pelt     # one detector
pytest -x --tb=short                # stop at the first failure
pytest -m reproduction              # reproduction tests (deselected by default)
pytest -m "not slow"                # without the slow tests (detection quality)
```

`pyproject.toml` sets the options (`--strict-markers`, `-m "not reproduction"`).
Tests whose optional dependency is not installed are skipped: the CI installs
`pip install -e .[dev,aeon,prophet,tglad,beast,changeforest]` and the CPU build of torch,
so it skips the TensorFlow-based detectors among others.
