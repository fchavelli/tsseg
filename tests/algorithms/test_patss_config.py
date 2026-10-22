"""PatssDetector's ``config``: missing keys filled from the defaults, in a copy.

``run_patss`` completes ``config`` with its default hyperparameters. It used to
do so in place: after ``predict`` the dictionary passed by the user (so also
``get_params()["config"]``) held every default, and its missing sections were
the module-level default dictionaries themselves. The pattern mining (``npbad``
and its Java miner) is replaced by a synthetic embedding, so these tests run
without ``npbad``.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from tsseg.algorithms.patss.algorithms import PaTSS_perso
from tsseg.algorithms.patss.detector import PatssDetector

pytestmark = pytest.mark.filterwarnings(
    "ignore:PatssDetector is deprecated:FutureWarning"
)

DEFAULTS = getattr(PaTSS_perso, "__DEFAULTS_HYPER_PARAMETERS")
N_TIMEPOINTS = 450
X = np.random.default_rng(0).normal(size=N_TIMEPOINTS)


@pytest.fixture
def mined_configs(monkeypatch):
    """Replace the pattern mining; record the config it receives."""
    configs = []

    def fake_mining(data, directory, attribute_id, interval, config):
        configs.append(config)
        # Three regimes; patterns 4k to 4k + 3 occur mostly in regime k.
        rng = np.random.default_rng(int(interval))
        regime = np.repeat(np.arange(3), len(data) // 3)
        frequent = np.arange(12)[:, None] // 4 == regime[None, :]
        embedding = (rng.random(frequent.shape) < np.where(frequent, 0.9, 0.1)) * 1.0
        return attribute_id, embedding, pd.DataFrame({"pattern": range(12)}), ""

    monkeypatch.setattr(
        PaTSS_perso, "__create_attribute_resolution_embedding", fake_mining
    )
    return configs


def test_predict_leaves_the_config_unchanged(mined_configs):
    config = {"segmentation": {"n_clusters": [2, 3]}}
    expected = copy.deepcopy(config)
    detector = PatssDetector(config=config)
    detector.fit_predict(X)
    assert config == expected
    assert detector.get_params()["config"] == expected


def test_partial_section_is_completed_from_the_defaults(mined_configs):
    config = {"pattern_mining": {"word_size": 8}}
    PatssDetector(config=config).fit_predict(X)
    assert config == {"pattern_mining": {"word_size": 8}}
    used = mined_configs[0]
    assert used["pattern_mining"] == {**DEFAULTS["pattern_mining"], "word_size": 8}
    assert used["pattern_filter"] == DEFAULTS["pattern_filter"]
    assert used["segmentation"] == DEFAULTS["segmentation"]


@pytest.mark.parametrize("config", [None, {"segmentation": {"n_clusters": [2]}}])
def test_run_does_not_share_the_default_dictionaries(mined_configs, config):
    defaults_before = copy.deepcopy(DEFAULTS)
    PatssDetector(config=config).fit_predict(X)
    used = mined_configs[0]
    assert used is not DEFAULTS
    for section in ("pattern_mining", "pattern_filter", "segmentation"):
        assert used[section] is not DEFAULTS[section]
    assert DEFAULTS == defaults_before


def test_documented_custom_config_changes_the_output(mined_configs):
    default_labels = PatssDetector().fit_predict(X)
    labels = PatssDetector(config={"segmentation": {"n_clusters": [2]}}).fit_predict(X)
    assert len(np.unique(default_labels)) == 3
    assert len(np.unique(labels)) == 2


@pytest.mark.parametrize("algorithm", ["logistic_regression", "gmm", "k_means"])
def test_every_segmentation_algorithm_returns_one_label_per_point(
    mined_configs, algorithm
):
    config = {"segmentation": {"algorithm": algorithm}}
    labels = PatssDetector(config=config).fit_predict(X)
    assert labels.shape == (N_TIMEPOINTS,)
    assert np.issubdtype(labels.dtype, np.integer)
    assert len(np.unique(labels)) == 3
