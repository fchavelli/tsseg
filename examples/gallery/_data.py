"""Load the gallery's numeric snapshots, offline, after checking their hashes."""

import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name):
    """Read a numeric snapshot after verifying its SHA-256 and row count."""
    manifest = json.loads((HERE / "data" / "manifest.json").read_text())
    record = manifest[name]
    path = HERE / "data" / record["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError(f"Snapshot checksum mismatch: {path}")
    frame = pd.read_csv(path)
    if len(frame) != record["rows"]:
        raise ValueError(f"Snapshot length mismatch: {path}")
    return frame
