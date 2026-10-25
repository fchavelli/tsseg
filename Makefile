.PHONY: help install install-conda clean clean-conda check-conda lint test docs

# Load .env file if it exists and export the variables
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Virtual environment created by make install (venv module of VENV_PYTHON).
VENV ?= .venv
# Interpreter that creates the virtual environment, e.g. VENV_PYTHON=python3.12.
VENV_PYTHON ?= python3
# Extras installed by make install and make install-conda. The all extra pins
# TensorFlow below 2.19, which has no wheel for Python 3.13 or later: there, pick
# other extras, e.g. EXTRAS=dev,torch,prophet,aeon,accelerators.
EXTRAS ?= all,dev
# Index from which make install takes torch first, e.g. the CPU build that the CI
# installs: TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu. Empty: torch
# comes from PyPI with the other dependencies (on Linux, with the CUDA libraries).
TORCH_INDEX_URL ?=

# If CONDA is not set in .env, try to find it in the PATH
ifeq ($(CONDA),)
    CONDA := $(shell which conda)
endif

# Name of the conda environment of make install-conda (set in environment.yml)
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
	@echo "  make install        Create a virtual environment in VENV (default .venv)"
	@echo "                      and install tsseg in it, editable, with EXTRAS"
	@echo "                      (default all,dev)."
	@echo "  make install-conda  Create the conda environment $(CONDA_ENV_NAME) from"
	@echo "                      environment.yml and install tsseg in it with EXTRAS."
	@echo "  make clean          Remove the virtual environment VENV."
	@echo "  make clean-conda    Remove the conda environment $(CONDA_ENV_NAME)."
	@echo "  make lint           Run the CI lint job (ruff check + ruff format --check)."
	@echo "  make test           Run the CI test job (long: select tests with"
	@echo "                      PYTEST_ARGS=\"tests/algorithms -k PeltDetector\")."
	@echo "  make docs           Run the CI docs build, warnings as errors, into"
	@echo "                      DOCS_BUILD_DIR (default docs/_build/html; needs the docs extra)."
	@echo ""
	@echo "Configuration:"
	@echo "  - VENV_PYTHON=python3.12 picks the interpreter of the virtual environment."
	@echo "  - TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu installs the CPU"
	@echo "    build of torch, as the CI does."
	@echo "  - test and docs run PYTHON (default python): activate the environment first."
	@echo "  - The conda targets find 'conda' in your PATH, or take it from a '.env'"
	@echo "    file with 'CONDA=/path/to/conda'."

install:
	@echo "--> Creating the virtual environment $(VENV) with $(VENV_PYTHON)..."
	"$(VENV_PYTHON)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --upgrade pip
ifneq ($(TORCH_INDEX_URL),)
	"$(VENV)/bin/python" -m pip install torch --index-url "$(TORCH_INDEX_URL)"
endif
	"$(VENV)/bin/python" -m pip install -e ".[$(EXTRAS)]"
	@echo "--> Installation complete."
	@echo "--> To activate the environment, run: source $(VENV)/bin/activate"

clean:
	@echo "--> Removing the virtual environment $(VENV)..."
	rm -rf "$(VENV)"
	@echo "--> Done."

check-conda:
	@if [ -z "$(CONDA)" ] || ! [ -x "$(CONDA)" ]; then \
		echo "Error: conda executable not found or not executable at '$(CONDA)'"; \
		echo "Please ensure conda is in your PATH, or create a .env file with the correct CONDA=/path/to/conda"; \
		exit 1; \
	fi

install-conda: check-conda
	@echo "--> Using conda at: $(CONDA)"
	@echo "--> Creating conda environment $(CONDA_ENV_NAME) from environment.yml..."
	"$(CONDA)" env create -f environment.yml || (echo "Conda env creation failed, maybe it already exists. Trying to update." && "$(CONDA)" env update -f environment.yml --prune)
	@echo "--> Installing tsseg in the conda environment..."
	@"$(CONDA)" run -n $(CONDA_ENV_NAME) pip install -e ".[$(EXTRAS)]"
	@echo "--> Installation complete."
	@echo "--> To activate the environment, run: conda activate $(CONDA_ENV_NAME)"

clean-conda: check-conda
	@echo "--> Removing conda environment $(CONDA_ENV_NAME)..."
	"$(CONDA)" env remove -n $(CONDA_ENV_NAME)
	@echo "--> Done."

# Same commands as the lint job of .github/workflows/ci.yml (ruff version pinned there
# and in the dev extra: pip install -e .[dev]).
lint:
	ruff check .
	ruff format --check .

# Same command as the test job of .github/workflows/ci.yml, which installs
# pip install -e .[dev,aeon,prophet,tglad,beast,changeforest,fluss]: tests whose
# optional dependency is missing are skipped, and pyproject.toml deselects the
# reproduction tests (make test PYTEST_ARGS="-m reproduction" runs them).
test:
	$(PYTHON) -m pytest --tb=short -q $(PYTEST_ARGS)

# Same command as the docs job of .github/workflows/ci.yml (-W: warnings are
# errors). -E rereads every source file, so that a rebuild in an existing
# directory reports the warnings of unchanged files, as the fresh CI build does.
docs:
	$(PYTHON) -m sphinx -W -E -b html docs $(DOCS_BUILD_DIR)
