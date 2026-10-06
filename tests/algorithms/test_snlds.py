"""Packaging and behavioural tests of SNLDSDetector."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tsseg.algorithms.snlds as snlds_pkg

VENDOR = Path(snlds_pkg.__file__).parent / "vendor"


def test_import_does_not_load_tensorflow():
    """The detector must be importable without TensorFlow installed."""
    code = (
        "import sys\n"
        "from tsseg.algorithms import SNLDSDetector\n"
        "SNLDSDetector(n_states=2)\n"
        "print('tensorflow' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_vendored_code_is_self_contained():
    """No absolute ``snlds`` import: the code must not depend on ``sys.path``."""
    for path in VENDOR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                assert not (node.module or "").startswith("snlds"), path.name
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert not any(n.startswith("snlds") for n in names), path.name
    assert "Apache License" in (VENDOR / "LICENSE").read_text()


@pytest.mark.parametrize("n_states", [2, 3])
def test_labels_follow_n_states(n_states):
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")
    pytest.importorskip("tf_keras")
    from tsseg.algorithms import SNLDSDetector

    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(0, 1, (100, 2)), rng.normal(4, 1, (100, 2))])
    labels = SNLDSDetector(
        n_states=n_states, n_train_steps=3, random_state=0
    ).fit_predict(X)
    assert labels.shape == (200,)
    assert labels.dtype == np.int64
    assert labels.min() >= 0 and labels.max() < n_states
