"""PatssDetector is deprecated."""

import pytest

from tsseg.algorithms import PatssDetector


def test_constructor_warns():
    with pytest.warns(FutureWarning, match="PatssDetector is deprecated"):
        PatssDetector()
