"""Build one frozen splits file for all datasets, with leakage control.

Rules (documented in the thesis data chapter):
1. MAESTRO keeps its official train / validation / test split.
2. ASAP performances that are MAESTRO recordings inherit the MAESTRO split, so beat labels
   never come from a MAESTRO test recording.
3. An ASAP *piece* that has any performance in MAESTRO test is excluded from beat training
   entirely (the same composition must not be learned and then tested).
4. POP909 uses a seeded song-level 80/10/10 split.
5. Noise and impulse-response files are split 80/20 by a stable hash of the file name, so
   MAESTRO-Wild test conditions never use training noise or rooms.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pandas as pd

SPLITS = ("train", "validation", "test")


def maestro_id(audio_filename: str) -> str:
    """Stable id for a MAESTRO performance: the audio file stem."""
    return Path(audio_filename).stem


def maestro_splits(maestro_csv: str | Path) -> dict[str, list[str]]:
    df = pd.read_csv(maestro_csv)
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for _, row in df.iterrows():
        out[row["split"]].append(maestro_id(row["audio_filename"]))
    return {k: sorted(v) for k, v in out.items()}


def _asap_maestro_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "maestro" in col.lower():
            return col
    return None


def asap_beat_splits(
    asap_metadata_csv: str | Path, maestro: dict[str, list[str]]
) -> tuple[dict[str, list[str]], list[str]]:
    """Assign ASAP performances that have MAESTRO audio to MAESTRO's split.

    Returns (splits of ASAP performance ids, list of excluded ASAP piece folders).
    ASAP performance id = path of its performance MIDI relative to the ASAP root.
    """
    df = pd.read_csv(asap_metadata_csv)
    col = _asap_maestro_column(df)
    if col is None:
        raise ValueError(
            f"No MAESTRO column in {asap_metadata_csv}; columns are {list(df.columns)}"
        )
    split_of = {mid: s for s, ids in maestro.items() for mid in ids}

    df = df[df[col].notna() & (df[col].astype(str).str.len() > 0)].copy()
    df["maestro_id"] = df[col].astype(str).map(lambda p: Path(p).stem)
    df["split"] = df["maestro_id"].map(split_of)
    df = df[df["split"].notna()]
    df["piece"] = df["midi_performance"].map(lambda p: str(Path(p).parent))

    test_pieces = set(df.loc[df["split"] == "test", "piece"])
    out: dict[str, list[str]] = {s: [] for s in SPLITS}
    for _, row in df.iterrows():
        if row["split"] != "test" and row["piece"] in test_pieces:
            continue  # rule 3
        out[row["split"]].append(row["midi_performance"])
    return {k: sorted(v) for k, v in out.items()}, sorted(test_pieces)


def song_level_split(
    song_ids: list[str], seed: int = 17, ratios: tuple[float, float] = (0.8, 0.1)
) -> dict[str, list[str]]:
    ids = sorted(song_ids)
    random.Random(seed).shuffle(ids)
    n_train = round(len(ids) * ratios[0])
    n_val = round(len(ids) * ratios[1])
    return {
        "train": sorted(ids[:n_train]),
        "validation": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def hash_split(name: str, test_fraction: float = 0.2) -> str:
    """Deterministic train/test assignment that does not depend on directory listing order."""
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "train"


def split_directory(root: str | Path, exts: tuple[str, ...] = (".wav", ".flac")) -> dict:
    root = Path(root)
    out: dict[str, list[str]] = {"train": [], "test": []}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            rel = str(p.relative_to(root))
            out[hash_split(rel)].append(rel)
    return out


def write_splits(path: str | Path, splits: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(splits, indent=1, sort_keys=True))
