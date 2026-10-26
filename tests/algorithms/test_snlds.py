"""Packaging and behavioural tests of SNLDSDetector."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tsseg.algorithms.snlds as snlds_pkg
from tsseg.algorithms import SNLDSDetector
from tsseg.algorithms._dependencies import missing_dependencies

VENDOR = Path(snlds_pkg.__file__).parent / "vendor"


def test_import_does_not_load_tensorflow():
    """Importing and constructing the detector does not load TensorFlow.

    Without the TensorFlow stack the detector still imports; construction then
    raises ``ModuleNotFoundError``, which names the ``snlds`` extra.
    """
    code = (
        "import sys\n"
        "from tsseg.algorithms import SNLDSDetector\n"
        "try:\n"
        "    SNLDSDetector(n_states=2)\n"
        "except ModuleNotFoundError as error:\n"
        "    print(error)\n"
        "else:\n"
        "    print('constructed')\n"
        "print('tensorflow' in sys.modules)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    constructed, loaded = out.stdout.strip().splitlines()
    if missing_dependencies(SNLDSDetector.get_class_tag("python_dependencies")):
        assert 'pip install "tsseg[snlds]"' in constructed
    else:
        assert constructed == "constructed"
    assert loaded == "False"


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


def _tf_stack():
    pytest.importorskip("tensorflow")
    pytest.importorskip("tensorflow_probability")
    pytest.importorskip("tf_keras")


def test_schedules_follow_the_paper_scaled():
    """Reacher schedule of the paper (300,000 steps) scaled to n_train_steps."""
    _tf_stack()
    from tsseg.algorithms import SNLDSDetector

    temperature, xent = SNLDSDetector(n_train_steps=600)._get_schedules()
    # weight: constant until 600 / 6 = 100, then x0.975 per 600 / 600 = 1 step
    assert float(xent(0)) == pytest.approx(1000.0)
    assert float(xent(100)) == pytest.approx(1000.0)
    assert float(xent(110)) == pytest.approx(1000.0 * 0.975**10, rel=1e-5)
    assert float(xent(599)) == pytest.approx(1000.0 * 0.975**499, rel=1e-4)
    # temperature: constant until 600 / 3 = 200, then decays down to 1
    assert float(temperature(199)) == pytest.approx(1000.0)
    assert float(temperature(210)) == pytest.approx(1000.0 * 0.975**10, rel=1e-5)
    assert float(temperature(599)) == pytest.approx(1.0)


def test_schedules_switched_off():
    _tf_stack()
    from tsseg.algorithms import SNLDSDetector

    temperature, xent = SNLDSDetector(
        n_train_steps=600, use_temperature_annealing=False, use_cross_entropy_reg=False
    )._get_schedules()
    assert float(temperature(0)) == 1.0 and float(xent(0)) == 0.0


def test_window_training_and_seeded_predict():
    _tf_stack()
    from tsseg.algorithms import SNLDSDetector

    rng = np.random.default_rng(1)
    X = np.concatenate([rng.normal(0, 1, (300, 2)), rng.normal(4, 1, (300, 2))])
    det = SNLDSDetector(
        n_states=2, n_train_steps=3, train_window=64, batch_size=4, random_state=0
    ).fit(X)
    a, b = det.predict(X), det.predict(X)
    assert a.shape == (600,)
    # seeded, but TensorFlow's sampling in a while loop is not bit-reproducible
    assert np.mean(a == b) > 0.99


def _fit_weights(seed_of_python_random):
    """Weights after one training step, with Python's ``random`` seeded first."""
    import random

    rng = np.random.default_rng(2)
    X = np.concatenate([rng.normal(0, 1, (100, 1)), rng.normal(3, 1, (100, 1))])
    random.seed(seed_of_python_random)
    det = SNLDSDetector(
        n_states=2, n_train_steps=1, train_window=64, batch_size=2, random_state=0
    ).fit(X)
    return [v.numpy() for v in det._model.trainable_variables]


def test_random_state_fixes_the_initial_weights():
    """Same ``random_state``, same model, whatever the state of ``random``.

    Keras 3 draws the seed of an initializer built without one from Python's
    ``random`` module: unless the detector seeds it, two fits with the same
    ``random_state`` start from different weights.
    """
    _tf_stack()
    a, b = _fit_weights(1), _fit_weights(2)
    assert len(a) == len(b)
    for u, v in zip(a, b, strict=True):
        # one Adam step of 1e-4 from the same weights: equal up to the
        # rounding of multithreaded TensorFlow ops
        np.testing.assert_allclose(u, v, rtol=0, atol=1e-6)


def test_fit_leaves_the_python_random_stream_alone():
    """Seeding the initial weights does not reseed the caller's ``random``."""
    _tf_stack()
    import random

    random.seed(3)
    expected = random.random()
    random.seed(3)
    X = np.random.default_rng(2).normal(0, 1, (100, 1))
    SNLDSDetector(
        n_states=2, n_train_steps=1, train_window=64, batch_size=2, random_state=0
    ).fit(X)
    assert random.random() == expected
