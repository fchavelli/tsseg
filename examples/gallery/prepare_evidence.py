"""Features and reference columns for the gallery snapshots.

Audio preparation requires librosa and pyedflib; replaying the bundled numeric
snapshots does not. References are retained in separate columns, never features.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import welch

from examples.gallery.prepare import tokens


def novel(cache):
    text = (cache / "jekyll.txt").read_text(encoding="utf-8-sig")
    start = re.search(r"(?m)^STORY OF THE DOOR\s*$", text).start()
    text = text[start : text.index("*** END OF THE PROJECT GUTENBERG")]
    titles = [
        "STORY OF THE DOOR",
        "SEARCH FOR MR. HYDE",
        "DR. JEKYLL WAS QUITE AT EASE",
        "THE CAREW MURDER CASE",
        "INCIDENT OF THE LETTER",
        "INCIDENT OF DR. LANYON",
        "INCIDENT AT THE WINDOW",
        "THE LAST NIGHT",
        "DR. LANYON’S NARRATIVE",
        "HENRY JEKYLL’S FULL STATEMENT OF THE CASE",
    ]
    headings = [
        re.search(r"(?m)^" + re.escape(title) + r"\s*$", text) for title in titles
    ]
    words, chapters, offsets = [], [], []
    for i, match in enumerate(headings):
        offsets.append(len(words))
        body = text[
            match.end() : headings[i + 1].start()
            if i + 1 < len(headings)
            else len(text)
        ]
        tok = tokens(body.replace("’", "'"))
        words.extend(tok)
        chapters.extend([i + 1] * len(tok))
    dictionaries = {
        "first_person": {
            "i",
            "me",
            "my",
            "mine",
            "myself",
            "i'm",
            "i've",
            "i'll",
            "i'd",
        },
        "third_person": {"he", "him", "his", "himself"},
    }
    rows = []
    for i in range(0, len(words) - 299, 300):
        chapter = chapters[i + 150]  # reference at the bin centre
        row = dict(
            x=i,
            label=f"word {i}",
            chapter=chapter,
            testimony=int(chapter >= 9),
            reference_start=offsets[8],
            jekyll_start=offsets[9],
        )
        row.update(
            {
                key: 100 * sum(w in lexicon for w in words[i : i + 300]) / 300
                for key, lexicon in dictionaries.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def speech(cache):
    import librosa

    order = ["jackson", "nicolas", "jackson", "theo", "nicolas", "theo"]
    ids = {name: i for i, name in enumerate(dict.fromkeys(order))}
    seen = dict.fromkeys(order, 0)
    rows = []
    for block, speaker in enumerate(order):
        for j in range(20):
            digit, take = (7 * j + 3) % 10, 5 + 2 * seen[speaker] + j // 10
            name = f"{digit}_{speaker}_{take}.wav"
            rate, raw = wavfile.read(cache / name)
            if rate != 8000 or raw.ndim != 1 or raw.dtype != np.int16:
                raise ValueError(f"Unexpected FSDD recording format: {name}")
            audio, _ = librosa.effects.trim(raw.astype(float) / 32768, top_db=30)
            # Omit coefficient zero (energy); one observation per utterance.
            mfcc = librosa.feature.mfcc(
                y=audio, sr=rate, n_mfcc=13, n_fft=256, hop_length=80, n_mels=24
            )[1:]
            row = dict(
                x=len(rows),
                label=name,
                block=block,
                speaker=ids[speaker],
                speaker_name=speaker,
                digit=digit,
                take=take,
            )
            row.update(
                {f"mfcc_{i + 1:02}": val for i, val in enumerate(mfcc.mean(axis=1))}
            )
            rows.append(row)
        seen[speaker] += 1
    return pd.DataFrame(rows)


def sleep(cache):
    import pyedflib

    reader = pyedflib.EdfReader(str(cache / "SC4001E0-PSG.edf"))
    try:
        start = pd.Timestamp(reader.getStartdatetime())
        # Clock-based crop fixed before reading the expert annotation file.
        lo = int((pd.Timestamp("1989-04-24 23:00") - start).total_seconds())
        hi = int((pd.Timestamp("1989-04-25 08:00") - start).total_seconds())
        times = np.arange(lo, hi, 30)
        frames = []
        for channel in [0, 1, 2]:
            if reader.getSampleFrequency(channel) != 100:
                raise ValueError("Expected 100 Hz EEG/EOG")
            signal = reader.readSignal(
                channel, start=lo * 100, n=(hi - lo) * 100
            ).reshape(-1, 3000)
            freq, power = welch(signal, fs=100, nperseg=400, axis=1)
            features = {}
            for name, low, high in [
                ("delta", 0.5, 4),
                ("theta", 4, 8),
                ("alpha", 8, 12),
                ("sigma", 12, 16),
                ("beta", 16, 30),
            ]:
                energy = power[:, (freq >= low) & (freq < high)].sum(axis=1)
                features[f"ch{channel}_{name}"] = np.log10(energy + 1e-10)
            frames.append(pd.DataFrame(features))
        if reader.getSampleFrequency(4) != 1:
            raise ValueError("Expected a 1 Hz recorded EMG envelope")
        emg = reader.readSignal(4, start=lo, n=hi - lo).reshape(-1, 30)
    finally:
        reader.close()
    frame = pd.concat(frames, axis=1)
    frame["emg"] = np.log10(np.maximum(emg.mean(axis=1), 1e-5))
    frame["x"] = (start + pd.to_timedelta(times, unit="s")).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    frame["label"] = frame.x
    reader = pyedflib.EdfReader(str(cache / "SC4001EC-Hypnogram.edf"))
    try:
        onsets, durations, descriptions = reader.readAnnotations()
    finally:
        reader.close()
    mapping = {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 3,
        "Sleep stage R": 4,
    }
    labels = np.full(len(frame), -1)
    for onset, duration, description in zip(
        onsets, durations, descriptions, strict=True
    ):
        labels[(times >= onset) & (times < onset + duration)] = mapping.get(
            description, -1
        )
    if (labels < 0).any():
        raise ValueError("Unscored/artifact epochs require an explicit exclusion rule")
    frame["stage"] = labels
    return frame


def annotations(cache):
    text = (cache / "chloroplast.gb").read_text()
    matches = list(re.finditer(r"^ {5}(\S+)\s+(.*)$", text, re.M))
    rows = []
    for i, match in enumerate(matches):
        if match[1] != "rRNA":
            continue
        location = re.fullmatch(r"(complement\()?([0-9]+)\.\.([0-9]+)\)?", match[2])
        if location is None:
            raise ValueError("Unsupported rRNA location")
        block = text[
            match.end() : matches[i + 1].start()
            if i + 1 < len(matches)
            else text.index("ORIGIN")
        ]
        product = re.search(r'/product="([^"]+)"', block)[1]
        rows.append(
            dict(
                x=int(location[2]),
                label=product,
                end=int(location[3]),
                strand=-1 if location[1] else 1,
            )
        )
    return pd.DataFrame(rows).sort_values("x").reset_index(drop=True)


def grid(cache):
    rows = []
    for name in [
        "grid.json",
        "generation_2024-01-15.json",
        "generation_2024-01-29.json",
    ]:
        for record in json.loads((cache / name).read_text())["data"]:
            row = dict(x=record["from"], label=record["from"])
            row.update({part["fuel"]: part["perc"] for part in record["generationmix"]})
            rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates("x").sort_values("x")
    frame = frame[(frame.x >= "2024-01-01T00:00Z") & (frame.x < "2024-02-01T00:00Z")]
    intensity = []
    for name in [
        "intensity.json",
        "intensity_2024-01-15.json",
        "intensity_2024-01-29.json",
    ]:
        intensity.extend(
            dict(x=r["from"], carbon_intensity=r["intensity"]["actual"])
            for r in json.loads((cache / name).read_text())["data"]
        )
    frame = frame.merge(
        pd.DataFrame(intensity).drop_duplicates("x"), on="x", validate="one_to_one"
    )
    expected = pd.date_range(
        "2024-01-01", "2024-02-01", freq="30min", inclusive="left", tz="UTC"
    )
    if (
        not pd.DatetimeIndex(pd.to_datetime(frame.x, utc=True)).equals(expected)
        or frame.isna().any().any()
    ):
        raise ValueError("Expected every half-hour in January 2024")
    return frame.reset_index(drop=True)
