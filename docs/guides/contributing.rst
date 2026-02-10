
Contributions
=============

We welcome contributions to ``tsseg``! Whether it's adding a new algorithm, improving documentation, or fixing bugs, your help is appreciated.

Development Setup
-----------------

1.  **Fork and Clone**:
    Fork the repository on GitHub and clone it locally.

    .. code-block:: bash

        git clone https://github.com/fchavell/tsseg.git
        cd tsseg

2.  **Install Development Dependencies**:
    We use ``ruff`` for linting/formatting and ``pytest`` for testing.

    .. code-block:: bash

        pip install -e .[dev]
        pre-commit install

Code Style
----------

This project adheres to strict code quality standards to maintain professional reliability.

-   **Linting**: use ``ruff check tsseg`` to catch errors.
-   **Formatting**: use ``ruff format tsseg`` to auto-format code (Black-compatible).
-   **Type Checking**: use ``mypy tsseg`` for static analysis.

Please ensure all checks pass before submitting a Pull Request.

Testing
-------

Run the test suite using pytest:

.. code-block:: bash

    pytest tests/

When adding a new feature, please include relevant tests in ``tests/``.
If adding a new algorithm, ensure it passes the common estimator checks (see :doc:`detectors`).
