# Makefile for building vendorized Rbeast C extension
#
# Usage:
#   make          # build the extension in-place
#   make clean    # remove build artifacts
#   make test     # quick smoke test
#
# Prerequisites: gcc, python3-dev, numpy (any version, including >=2.0)
#
# The compiled .so will be placed in py_src/Rbeast/ so that
# ``import Rbeast`` works when py_src/ is on PYTHONPATH.

PYTHON ?= python3

.PHONY: all clean test

all:
	cd $(CURDIR) && $(PYTHON) setup.py build_ext --inplace

clean:
	rm -rf build/
	find py_src/Rbeast -name "*.so" -delete 2>/dev/null || true
	find py_src/Rbeast -name "*.pyd" -delete 2>/dev/null || true

test:
	cd $(CURDIR) && PYTHONPATH=py_src:$$PYTHONPATH $(PYTHON) -c \
		"import Rbeast as rb; import numpy as np; \
		 o = rb.beast(np.random.randn(200), season='none', quiet=True); \
		 print('OK – ncp_median =', o.trend.ncp_median)"
