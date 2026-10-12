.PHONY: help install clean lint test docs

# Load .env file if it exists and export the variables
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# If CONDA is not set in .env, try to find it in the PATH
ifeq ($(CONDA),)
    CONDA := $(shell which conda)
endif

# Default environment name
CONDA_ENV_NAME = tsseg-env

# Interpreter of the environment where tsseg is installed (used by test and docs).
PYTHON ?= python
# Extra pytest arguments, e.g. make test PYTEST_ARGS="tests/algorithms -k PeltDetector"
PYTEST_ARGS ?=
# Output directory of make docs (the CI builds into docs/_build/html).
DOCS_BUILD_DIR ?= docs/_build/html

help:
	@echo "Makefile for tsseg"
	@echo ""
	@echo "Usage:"
	@echo "  make install    Create conda environment and install tsseg."
	@echo "  make clean      Remove the conda environment."
	@echo "  make lint       Run the CI lint job (ruff check + ruff format --check)."
	@echo "  make test       Run the CI test job (long: select tests with"
	@echo "                  PYTEST_ARGS=\"tests/algorithms -k PeltDetector\")."
	@echo "  make docs       Run the CI docs build, warnings as errors, into"
	@echo "                  DOCS_BUILD_DIR (default docs/_build/html; needs the docs extra)."
	@echo ""
	@echo "Configuration:"
	@echo "  - The Makefile will automatically find 'conda' in your PATH."
	@echo "  - Alternatively, create a '.env' file with 'CONDA=/path/to/conda' to specify the path."

install:
	@echo "--> Checking for conda..."
	@if [ -z "$(CONDA)" ] || ! [ -x "$(CONDA)" ]; then \
		echo "Error: conda executable not found or not executable at '$(CONDA)'"; \
		echo "Please ensure conda is in your PATH, or create a .env file with the correct CONDA=/path/to/conda"; \
		exit 1; \
	fi
	@echo "--> Using conda at: $(CONDA)"
	@echo "--> Creating conda environment $(CONDA_ENV_NAME) from environment.yml..."
	"$(CONDA)" env create -f environment.yml || (echo "Conda env creation failed, maybe it already exists. Trying to update." && "$(CONDA)" env update -f environment.yml --prune)
	@echo "--> Activating conda environment and installing tsseg..."
	@"$(CONDA)" run -n $(CONDA_ENV_NAME) pip install -e .[all]
	@echo "--> Installation complete."
	@echo "--> To activate the environment, run: conda activate $(CONDA_ENV_NAME)"

clean:
	@echo "--> Removing conda environment $(CONDA_ENV_NAME)..."
	@if [ -z "$(CONDA)" ] || ! [ -x "$(CONDA)" ]; then \
		echo "Error: conda executable not found or not executable at '$(CONDA)'"; \
		echo "Please ensure conda is in your PATH, or create a .env file with the correct CONDA=/path/to/conda"; \
		exit 1; \
	fi
	"$(CONDA)" env remove -n $(CONDA_ENV_NAME)
	@echo "--> Done."

# Same commands as the lint job of .github/workflows/ci.yml (ruff version pinned there
# and in the dev extra: pip install -e .[dev]).
lint:
	ruff check .
	ruff format --check .

# Same command as the test job of .github/workflows/ci.yml, which installs
# pip install -e .[dev,aeon,prophet,tglad,beast]: tests whose optional dependency
# is missing are skipped, and pyproject.toml deselects the reproduction tests
# (make test PYTEST_ARGS="-m reproduction" runs them).
test:
	$(PYTHON) -m pytest --tb=short -q $(PYTEST_ARGS)

# Same command as the docs job of .github/workflows/ci.yml (-W: warnings are
# errors). -E rereads every source file, so that a rebuild in an existing
# directory reports the warnings of unchanged files, as the fresh CI build does.
docs:
	$(PYTHON) -m sphinx -W -E -b html docs $(DOCS_BUILD_DIR)
