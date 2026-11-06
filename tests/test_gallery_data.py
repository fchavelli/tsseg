"""Integrity and preprocessing of the example gallery's numeric snapshots.

These checks use only core dependencies and never download original sources.
The fits themselves are replayed by ``python -m examples.gallery.run_all``.
"""

import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest

from examples.gallery import _data, prepare


@pytest.mark.parametrize(
    "name",
    [
        "novel",
        "dna",
        "tides",
        "climate_titles",
        "grid",
        "speech",
        "sleep",
        "annotations",
    ],
)
def test_snapshot_integrity_and_order(name):
    frame = _data.load(name)
    groups = frame.groupby("group") if "group" in frame else [(name, frame)]
    for _, group in groups:
        assert group.x.is_monotonic_increasing
        assert group.x.is_unique
        assert not group.isna().any().any()


def test_changed_snapshot_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    content = b"x,label,value\n0,zero,1\n"
    (tmp_path / "data" / "demo.csv").write_bytes(content + b"1,one,2\n")
    manifest = {
        "demo": {
            "file": "demo.csv",
            "rows": 1,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    }
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(_data, "HERE", tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _data.load("demo")


def test_changed_download_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prepare, "urlopen", lambda *a, **kw: io.BytesIO(b"revised data")
    )
    source = {
        "url": "https://example.org/data",
        "sha256": hashlib.sha256(b"original").hexdigest(),
    }
    with pytest.raises(ValueError, match="Upstream bytes changed"):
        prepare.fetch(tmp_path, "raw.csv", source)
    assert not (tmp_path / "raw.csv").exists()


def test_tide_join_must_not_hide_a_missing_hour(tmp_path):
    observed = [
        {"t": "2012-10-15 00:00", "v": "1.5"},
        {"t": "2012-10-15 01:00", "v": "1.4"},
    ]
    predicted = [{"t": "2012-10-15 00:00", "v": "1.2"}]
    (tmp_path / "water.json").write_text(json.dumps({"data": observed}))
    (tmp_path / "predictions.json").write_text(json.dumps({"predictions": predicted}))
    with pytest.raises(ValueError, match="every expected hour"):
        prepare.tides(tmp_path)


def test_physical_units_and_denominators():
    dna = _data.load("dna")
    assert np.allclose(dna[["a", "c", "g", "t"]].sum(axis=1), 1)
    assert np.allclose(dna.gc, 100 * (dna.g + dna.c))
    assert dna.x.iloc[-1] + 399 == 154400  # final incomplete window omitted
    tides = _data.load("tides")
    assert np.allclose(tides.residual, tides.observed - tides.predicted)
    assert len(tides) == 29 * 24
    titles = _data.load("climate_titles")
    assert np.allclose(titles.rate, 1000 * titles["count"] / titles.climate_titles)
    assert np.allclose(titles.log_rate, np.log1p(titles.rate))


def test_climate_titles_cover_every_year_for_each_expression():
    frame = _data.load("climate_titles")
    assert list(frame.group.unique()) == list(prepare.CLIMATE_EXPRESSIONS)
    for _, series in frame.groupby("group"):
        assert series.x.tolist() == list(range(1990, 2026))
    # One denominator per year, shared by all expressions.
    assert frame.groupby("x").climate_titles.nunique().eq(1).all()


def test_speech_has_balanced_content_and_no_repeated_recordings():
    frame = _data.load("speech")
    assert frame.label.is_unique
    assert len(frame) == 120
    for _, block in frame.groupby("block"):
        assert block.speaker.nunique() == 1
        assert block.digit.value_counts().to_dict() == dict.fromkeys(range(10), 2)
    assert frame.groupby("speaker").block.nunique().eq(2).all()


def test_expert_sleep_labels_align_with_fixed_clock_epochs():
    frame = _data.load("sleep")
    expected = pd.date_range(
        "1989-04-24 23:00", "1989-04-25 08:00", freq="30s", inclusive="left"
    )
    assert pd.DatetimeIndex(pd.to_datetime(frame.x)).equals(expected)
    assert set(frame.stage) == {0, 1, 2, 3, 4}


def test_saved_fits_exclude_annotation_columns():
    figures = _data.HERE.parents[1] / "docs/examples/_figures"
    reference_columns = {
        "testimony",
        "chapter",
        "reference_start",
        "jekyll_start",
        "speaker",
        "speaker_name",
        "digit",
        "take",
        "block",
        "stage",
        "activity",
        "activity_name",
        "carbon_intensity",
        "count",
        "climate_titles",
    }
    from examples.gallery.run_all import MODULES

    for slug in MODULES:
        record = json.loads((figures / f"{slug}.json").read_text())
        assert not reference_columns.intersection(record["features"])
