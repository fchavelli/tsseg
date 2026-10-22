"""PatssDetector writes nothing on the standard output (``npbad`` required)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("npbad")

from tsseg.algorithms import PatssDetector  # noqa: E402


def test_predict_prints_nothing(capsys):
    rng = np.random.default_rng(0)
    X = np.concatenate([np.sin(np.arange(200) / 3.0), rng.normal(0.0, 1.0, 200)])
    with pytest.warns(FutureWarning, match="deprecated"):
        detector = PatssDetector()
    labels = detector.fit_predict(X)
    assert labels.shape == X.shape
    assert capsys.readouterr().out == ""
