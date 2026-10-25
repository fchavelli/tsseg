:orphan:

Contributions
=============

We welcome contributions to ``tsseg``! Whether it's adding a new algorithm, improving documentation, or fixing bugs, your help is appreciated.

Development Setup
-----------------

1.  **Fork and Clone**:
    Fork the repository on GitHub and clone it locally.

    .. code-block:: bash

        git clone https://github.com/fchavelli/tsseg.git
        cd tsseg

2.  **Install Development Dependencies**:
    We use ``ruff`` for linting/formatting and ``pytest`` for testing, both in
    the ``dev`` extra. In a virtual environment:

    .. code-block:: bash

        python -m venv .venv
        source .venv/bin/activate
        pip install -e ".[all,dev]"

    ``make install`` runs these steps (``make help`` lists its options:
    ``EXTRAS``, ``VENV``, ``VENV_PYTHON``, ``TORCH_INDEX_URL``);
    ``make install-conda`` creates the conda environment ``tsseg-env`` from
    ``environment.yml`` instead. To install what the CI installs, with the CPU
    build of torch:

    .. code-block:: bash

        pip install torch --index-url https://download.pytorch.org/whl/cpu
        pip install -e ".[dev,aeon,prophet,tglad,beast,changeforest,fluss]"

Code Style
----------

This project adheres to strict code quality standards to maintain professional reliability.

-   **Linting**: use ``ruff check tsseg`` to catch errors.
-   **Formatting**: use ``ruff format tsseg`` to auto-format code (Black-compatible).

You can also run ``make lint`` which combines both. Please ensure all checks
pass before submitting a Pull Request.

Testing
-------

Run the test suite using pytest:

.. code-block:: bash

    pytest tests/

``make test`` runs the same command as the test job of the CI
(``pytest --tb=short -q``). The full suite takes a long time; pass a selection
in ``PYTEST_ARGS``:

.. code-block:: bash

    make test PYTEST_ARGS="tests/algorithms -k PeltDetector"

Tests whose optional dependency is not installed are skipped. The CI installs
``pip install -e .[dev,aeon,prophet,tglad,beast,changeforest,fluss]``, so it
skips the TensorFlow-based detectors among others.

When adding a new feature, please include relevant tests in ``tests/``.
If adding a new algorithm, ensure it passes the common estimator checks (see :doc:`../detectors/index`).

Algorithm test suite
^^^^^^^^^^^^^^^^^^^^

The ``tests/algorithms/`` directory contains a scikit-learn-style test suite
that **automatically discovers** every algorithm listed in
``tsseg.algorithms.__all__``.  Adding a new detector to ``__all__`` is
sufficient for it to be exercised by all generic checks — no manual test
registration required.

The generic checks are organised into six modules:

``test_common.py`` — **Estimator contract** (scikit-learn style)
    Verifies the generic API contract that every ``BaseSegmenter`` must honour:

    * ``get_params`` / ``set_params`` round-trip
    * ``clone`` produces an equivalent, independent estimator
    * ``fit`` returns ``self``
    * ``is_fitted`` lifecycle (``False`` → ``True``)
    * ``NotFittedError`` when predicting before fitting
    * ``reset`` clears fitted state
    * ``repr`` does not crash
    * Pickle round-trip (unless ``cant_pickle=True``)

``test_tags.py`` — **Tag contract**
    Ensures every algorithm declares valid metadata:

    * ``returns_dense`` (bool), ``detector_type`` (known value)
    * At least one ``capability:*`` tag is ``True``
    * ``returns_dense`` is consistent with ``detector_type``

``test_predict.py`` — **Prediction output contract**
    Validates that ``fit_predict`` output conforms to declared tags:

    * Change-point detectors (``returns_dense=True``): 1-D strictly increasing
      array of indices in ``[1, N - 1]``
    * State detectors: array of integer labels of length ``N``
    * ``fit_predict(X)`` equals ``fit(X); predict(X)`` for deterministic algorithms
    * Repeated calls produce identical results (determinism check)
    * Both univariate and multivariate inputs are exercised

``test_input_validation.py`` — **Robustness**
    Checks that detectors raise clear errors on bad inputs:

    * Invalid types (lists, strings), empty arrays
    * Multivariate data when ``capability:multivariate`` is ``False``
    * NaN values when ``capability:missing_values`` is ``False``
    * ``pd.Series`` and ``pd.DataFrame`` are accepted

``test_docstrings.py`` — **Docstring contract**
    The ``Parameters`` section of the class docstring lists exactly the
    constructor parameters, and the defaults it states are the actual ones.

``test_guided_no_change_point.py`` — **Guided mode**
    ``apply_supervision`` (in ``conftest.py``) sets the count of every
    detector tagged ``capability:semi_supervised``, and a guidance without
    any change point (``n_cps=0``, ``n_segments=1``, ``n_states=1``) passes
    the parameter validation; the module lists the detectors that still
    refuse it as expected failures.

Running a single algorithm:

.. code-block:: bash

    pytest tests/algorithms/ -k PeltDetector -v

Running only the fast contract checks (no fit/predict):

.. code-block:: bash

    pytest tests/algorithms/test_tags.py tests/algorithms/test_docstrings.py -v

If your algorithm requires non-default constructor arguments or optional
dependencies, add an entry to ``ALGORITHM_OVERRIDES`` in
``tests/algorithms/conftest.py``. If it has a guided mode and takes the number
of change points, segments or states under a new parameter name, add the name
to ``SUPERVISION_PARAMS`` in the same file. ``tests/README.md`` describes these
helpers.

A detector with optional dependencies lists their top-level modules in its
``python_dependencies`` tag; ``BaseSegmenter.__init__`` checks them at
construction. Its module must import without them
(``tests/test_optional_dependencies.py`` checks it): import them inside the
methods that use them, and move code that needs them at definition time
(numba kernels, ``torch.nn.Module`` subclasses) to a private sub-module. A
module installed by an extra not yet in ``EXTRAS``
(``tsseg/algorithms/_dependencies.py``) is added there, so that the error
message names the extra.

Reproduction tests
^^^^^^^^^^^^^^^^^^

Tests that reproduce a published result live in ``tests/reproduction/``.
They run a detector on a dataset used by the publication of its method,
downloaded on demand from the original source and pinned by its SHA-256
digest. They are marked ``reproduction`` and deselected by default, so they
do not run in CI; each one has a fast synthetic counterpart that does.

.. code-block:: bash

    pytest -m reproduction

The protocol (data helper, reference values, tolerances) is described in
``tests/reproduction/README.md``.

Documentation
-------------

The documentation is built with Sphinx (``pip install -e .[docs]``).
``make docs`` runs the same build as the docs job of the CI, where any
warning is an error, and writes the HTML pages to ``docs/_build/html``:

.. code-block:: bash

    make docs
    make docs DOCS_BUILD_DIR=/tmp/tsseg-docs   # another output directory

It rereads every source file (``-E``), so that a rebuild in an existing
directory reports the warnings of unchanged files, as the fresh CI build does.

Packaging
---------

The packages published on PyPI are built by the CI, from a fresh checkout,
when a ``v*`` tag is pushed. A build from a working copy can ship more:
setuptools packages every ``tsseg*`` package it finds on disk, git-ignored
folders included. To see what a release of a commit contains, build it from
an export of that commit (``build`` is in the ``dev`` extra):

.. code-block:: bash

    src=$(mktemp -d)
    git archive HEAD | tar -x -C "$src"
    python -m build "$src" --outdir dist

Besides the code, the wheel carries the data files that
``[tool.setuptools.package-data]`` selects in ``pyproject.toml``: the MoCap
trials of ``tsseg/data`` and the license file of each vendored component.
README files and MATLAB files stay out. ``tests/test_packaging.py`` checks
that every license file under ``tsseg/`` is selected.
