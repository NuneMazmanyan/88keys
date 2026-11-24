"""Build the MAESTRO-Wild test set from preprocessed MAESTRO test pieces.

Output (data/processed/maestro_wild by default):
  <condition>/<id>.flac     degraded audio, one folder per condition
  labels/<id>.npz           note + pedal labels (cropped if --max-seconds is used)
  manifest.csv              id, duration, n_notes
  conditions.csv            what each condition does, and which IR / noise files exist

  python scripts/build_wild.py                     # all available test pieces, full length
  python scripts/build_wild.py --max-seconds 60    # quick version: first 60 s of each piece
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from pianoscribe.data.audio import SAMPLE_RATE, load_audio, save_flac
from pianoscribe.data.notes import Performance, crop
from pianoscribe.eval import wild


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maestro", type=Path, default=Path("data/processed/maestro_16k"))
    ap.add_argument("--rirs", type=Path, default=Path("data/raw/rirs_noises/RIRS_NOISES"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/maestro_wild"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--max-pieces", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=[c.name for c in wild.CONDITIONS])
    args = ap.parse_args()

    manifest = pd.read_csv(args.maestro / "manifest.csv")
    pieces = manifest[manifest["split"] == args.split].sort_values("id")
    if args.max_pieces:
        pieces = pieces.head(args.max_pieces)
    if pieces.empty:
        raise SystemExit(f"No '{args.split}' pieces in {args.maestro}/manifest.csv")

    conds = [wild.CONDITION_BY_NAME[c] for c in args.conditions]
    irs = noises = None
    if any(c.reverb or c.snr_db is not None for c in conds):
        if not args.rirs.exists():
            raise SystemExit(f"Missing {args.rirs}. Run scripts/download_rirs_noises.py first.")
        cat = wild.catalog(args.rirs)
        irs = wild.ir_bank(args.rirs, cat["irs"]["test"])
        noises = [wild.load_mono(args.rirs / f) for f in cat["noises"]["test"]]
        noises = [n for n in noises if len(n) > SAMPLE_RATE]  # skip clips under 1 s
        print(
            f"Test IRs: {len(irs['medium'])} medium, {len(irs['long'])} long; "
            f"test noises: {len(noises)}"
        )

    rows = []
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    for _, row in tqdm(list(pieces.iterrows()), desc="pieces"):
        pid = row["id"]
        audio = load_audio(args.maestro / "audio" / f"{pid}.flac")
        perf = Performance.load(args.maestro / "labels" / f"{pid}.npz")
        if args.max_seconds:
            audio = audio[: int(args.max_seconds * SAMPLE_RATE)]
            perf = crop(perf, len(audio) / SAMPLE_RATE)
        perf.save(args.out / "labels" / f"{pid}.npz")
        for cond in conds:
            dst = args.out / cond.name / f"{pid}.flac"
            if not dst.exists():
                save_flac(dst, wild.degrade(audio, cond, pid, irs, noises))
        rows.append({"id": pid, "duration": len(audio) / SAMPLE_RATE, "n_notes": len(perf.notes)})

    pd.DataFrame(rows).to_csv(args.out / "manifest.csv", index=False)
    pd.DataFrame([{"condition": c.name, "description": c.description} for c in conds]).to_csv(
        args.out / "conditions.csv", index=False
    )
    print(f"MAESTRO-Wild: {len(rows)} pieces x {len(conds)} conditions -> {args.out}")


if __name__ == "__main__":
    main()
