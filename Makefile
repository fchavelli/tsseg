.PHONY: help install clean

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

help:
	@echo "Makefile for tsseg"
	@echo ""
	@echo "Usage:"
	@echo "  make install    Create conda environment and install tsseg."
	@echo "  make clean      Remove the conda environment."
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
	@echo "--> Running autopatch script..."
	@"$(CONDA)" run -n $(CONDA_ENV_NAME) python install_autopatch.py
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