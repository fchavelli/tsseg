import os
import pooch
import numpy as np
import pandas as pd

from importlib.resources import files

# 1. Create a Pooch instance for remote data fetching.
CACHE_PATH = pooch.os_cache("tsseg")
VERSION = "v1"
GOODBOY = pooch.create(
    path=CACHE_PATH,
    base_url="https://raw.githubusercontent.com/your_username/your_data_repo/main/",
    version=VERSION,
    version_dev="main",
    env="TSSEG_DATA_DIR",
)

# 2. Define a registry of files to download.
GOODBOY.registry = {
    # "UTSA.zip": "sha256:...",
    # "TSSB.zip": "sha256:...",
    # "has2023_master.csv.zip": "sha256:...",
}


# --- Internal utility functions ---

def _get_data_path():
    """Returns the path to the data directory within the package."""
    return files('tsseg.data')


# --- Public API ---

def fetch_dataset(filename, unzip=False):
    """
    Downloads and caches a dataset from the remote repository using Pooch.

    Parameters
    ----------
    filename : str
        The name of the file to download (must be in the Pooch registry).
    unzip : bool
        If True, unzips the archive and returns the path to the directory.

    Returns
    -------
    path : str
        Local path to the data file or the unzipped directory.
    """
    if filename not in GOODBOY.registry:
        raise ValueError(f"Dataset '{filename}' is not known. "
                         f"Available files: {list(GOODBOY.registry.keys())}")

    processor = pooch.Unzip() if unzip else None
    fname = GOODBOY.fetch(filename, processor=processor, progressbar=True)
    return fname


def load_local_dataset(filename):
    """
    Loads a small dataset included directly with the library.

    Parameters
    ----------
    filename : str
        The name of the file (e.g., 'my_data.csv') located in the `data` directory,
        including any subdirectories (e.g., 'mocap/86_01.csv').

    Returns
    -------
    data : pandas.DataFrame
        The loaded dataset.
    """
    data_path = _get_data_path()
    file_path = data_path.joinpath(filename)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file '{filename}' not found in the package data directory '{data_path}'. "
            "Verify the file path and your package installation."
        )
    return pd.read_csv(file_path, header=None)


# --- MoCap Dataset Loader ---
MOCAP_TRIALS = [f"86_{id}.csv" for id in ['01', '02', '03', '07', '08', '09', '10', '11', '14']]
MOCAP_COLUMN_NAMES = ['rhumerus_0', 'lhumerus_0', 'rfemur_0', 'lfemur_0', 'label']

def load_mocap(trial=0, return_X_y=True):
    """
    Loads a trial from the MoCap dataset and returns (X, y) arrays compatible with aeon/sklearn.

    Parameters
    ----------
    trial : int or str, default=0
        The identifier for the trial to load.
        - If int, it's the index in the list of available trials.
        - If str, it's the trial ID (e.g., '01', '07').
    return_X_y : bool, default=True
        If True, returns (X, y) arrays. If False, returns the full DataFrame.

    Returns
    -------
    X : np.ndarray, shape (n_timestamps, n_channels)
        The time series data (features only).
    y : np.ndarray, shape (n_timestamps,)
        The state labels for each timestamp.
    """
    if isinstance(trial, int):
        try:
            filename = MOCAP_TRIALS[trial]
        except IndexError:
            raise ValueError(f"Invalid trial index: {trial}. "
                             f"Must be between 0 and {len(MOCAP_TRIALS) - 1}.")
    elif isinstance(trial, str):
        trial_id = f"86_{trial}.csv"
        if trial_id not in MOCAP_TRIALS:
            raise ValueError(f"Invalid trial ID: '{trial}'. "
                             f"Available trials: {[t.split('_')[1].split('.')[0] for t in MOCAP_TRIALS]}")
        filename = trial_id
    else:
        raise TypeError(f"Type of 'trial' must be int or str, not {type(trial)}")

    # Prepend the subdirectory for mocap data
    full_path = os.path.join("mocap", filename)
    data = load_local_dataset(full_path)
    data.columns = MOCAP_COLUMN_NAMES

    if return_X_y:
        X = data.drop(columns=["label"]).to_numpy()
        y = data["label"].to_numpy()
        return X, y
    else:
        return data