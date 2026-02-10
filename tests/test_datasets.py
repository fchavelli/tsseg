"""Tests for tsseg.data.datasets — dataset loading utilities."""

import pytest
import numpy as np
import pandas as pd

from tsseg.data.datasets import load_mocap


# ---------------------------------------------------------------------------
# MoCap dataset (bundled with the package)
# ---------------------------------------------------------------------------

class TestLoadMocap:
    def test_default_return_X_y(self):
        """Default call returns (X, y) with correct shapes."""
        X, y = load_mocap()
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2, "X should be 2-D (n_timestamps, n_channels)"
        assert y.ndim == 1, "y should be 1-D (n_timestamps,)"
        assert X.shape[0] == y.shape[0], "X and y must have the same length"
        assert X.shape[1] == 4, "MoCap data has 4 channels"

    def test_return_dataframe(self):
        """return_X_y=False gives a DataFrame with the expected columns."""
        df = load_mocap(return_X_y=False)
        assert isinstance(df, pd.DataFrame)
        expected = ["rhumerus_0", "lhumerus_0", "rfemur_0", "lfemur_0", "label"]
        assert list(df.columns) == expected

    @pytest.mark.parametrize("trial_ref", [1, "07"])
    def test_load_by_index_and_id(self, trial_ref):
        """Can load by integer index or string trial ID."""
        X, y = load_mocap(trial=trial_ref)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError, match="Invalid trial index"):
            load_mocap(trial=99)

    def test_invalid_id_raises(self):
        with pytest.raises(ValueError, match="Invalid trial ID"):
            load_mocap(trial="99")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            load_mocap(trial=None)

    def test_trials_are_distinct(self):
        """Different trial indices return different data."""
        X0, _ = load_mocap(trial=0)
        X1, _ = load_mocap(trial=1)
        assert not np.array_equal(X0, X1)
