"""PatssDetector is deprecated."""

import pytest

from tsseg.algorithms import PatssDetector


def test_constructor_warns(monkeypatch):
    # Skip the dependency check at construction: npbad need not be installed.
    monkeypatch.setattr(
        PatssDetector, "_tags", {**PatssDetector._tags, "python_dependencies": None}
    )
    with pytest.warns(FutureWarning, match="PatssDetector is deprecated"):
        PatssDetector()
