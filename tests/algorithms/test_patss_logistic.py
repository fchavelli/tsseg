"""PaTSS's logistic-regression segmentation step across scikit-learn versions.

PaTSS fits a one-vs-rest elastic-net logistic regression on the pattern-based
embedding. scikit-learn removed ``LogisticRegression(multi_class=...)`` in 1.8,
which made ``PatssDetector`` fail with a ``TypeError``. These tests exercise the
segmentation step directly, so they run without ``npbad`` (pattern mining).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsseg.algorithms.patss import _sklearn_compat
from tsseg.algorithms.patss.algorithms import PaTSS_perso


def _embedding(seed: int = 0) -> dict[str, np.ndarray]:
    """Binary pattern embedding (patterns x time) with three regimes."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(mean, 1.0, size=(12, 150)) for mean in (-1.0, 0.0, 1.5)]
    return {"attribute": (np.hstack(blocks) > 0.3).astype(float)}


def test_logistic_regression_segmentation_runs():
    segment = getattr(PaTSS_perso, "__segment_embedding")
    parameters = {
        "n_clusters": [2, 3],
        "algorithm": "logistic_regression",
        "regularization": 1.0,
    }
    probabilities = segment(_embedding(), parameters)
    assert probabilities.ndim == 2
    assert probabilities.shape[1] == 450
    np.testing.assert_allclose(probabilities.sum(axis=0), 1.0)


def test_ovr_logistic_regression_is_one_vs_rest_elastic_net():
    model = _sklearn_compat.ovr_elasticnet_logistic_regression(C=2.0)
    estimator = model.estimator
    assert estimator.C == 2.0
    assert estimator.solver == "saga"
    assert estimator.l1_ratio == 0.5
    assert estimator.random_state == 0


def test_ovr_logistic_regression_raises_no_sklearn_deprecation():
    embedding = _embedding()["attribute"].T
    labels = np.repeat([0, 1, 2], 150)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = _sklearn_compat.ovr_elasticnet_logistic_regression(C=1.0)
        model.fit(embedding, labels)
    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "multi_class" in m or "penalty" in m]
    assert model.predict_proba(embedding).shape == (450, 3)


@pytest.mark.parametrize("n_classes", [2, 3])
def test_probabilities_sum_to_one(n_classes):
    embedding = _embedding()["attribute"].T
    labels = np.repeat(np.arange(3), 150) % n_classes
    model = _sklearn_compat.ovr_elasticnet_logistic_regression(C=1.0)
    probabilities = model.fit(embedding, labels).predict_proba(embedding)
    assert probabilities.shape == (450, n_classes)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
