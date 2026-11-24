"""Convert MAESTRO to the training format.

For every performance whose files are present:
  <out>/audio/<id>.flac    16 kHz mono FLAC
  <out>/labels/<id>.npz    notes (onset, offset, pitch, velocity) + sustain-pedal intervals
  <out>/manifest.csv       id, split, composer, title, duration, n_notes

Works on the full dataset or on a subset from download_maestro.py --subset.

  python scripts/preprocess_maestro.py --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from pianoscribe.data.audio import convert_to_canonical
from pianoscribe.data.notes import performance_from_midi
from pianoscribe.data.splits import maestro_id


def process_one(raw_root: str, out_root: str, row: dict) -> dict:
    raw, out = Path(raw_root), Path(out_root)
    pid = maestro_id(row["audio_filename"])
    audio_dst = out / "audio" / f"{pid}.flac"
    label_dst = out / "labels" / f"{pid}.npz"
    if not audio_dst.exists():
        duration = convert_to_canonical(raw / row["audio_filename"], audio_dst)
    else:
        duration = sf.info(str(audio_dst)).duration
    perf = performance_from_midi(raw / row["midi_filename"])
    label_dst.parent.mkdir(parents=True, exist_ok=True)
    perf.save(label_dst)
    return {
        "id": pid,
        "split": row["split"],
        "composer": row["canonical_composer"],
        "title": row["canonical_title"],
        "duration": round(float(duration), 3),
        "n_notes": len(perf.notes),
        "n_pedal": len(perf.pedal),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/maestro-v3.0.0"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/maestro_16k"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.raw / "maestro-v3.0.0.csv")
    present = df[
        df["audio_filename"].map(lambda f: (args.raw / f).exists())
        & df["midi_filename"].map(lambda f: (args.raw / f).exists())
    ]
    print(f"{len(present)} of {len(df)} performances present")
    if present.empty:
        raise SystemExit("Nothing to do. Run scripts/download_maestro.py first.")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_one, str(args.raw), str(args.out), r)
            for r in present.to_dict("records")
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="preprocess"):
            rows.append(fut.result())

    manifest = pd.DataFrame(rows).sort_values(["split", "id"])
    args.out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out / "manifest.csv", index=False)
    summary = manifest.groupby("split").agg(pieces=("id", "count"), hours=("duration", "sum"))
    summary["hours"] = (summary["hours"] / 3600).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
