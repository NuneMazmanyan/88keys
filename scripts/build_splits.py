"""Write data/processed/splits.json for every dataset that is present.

  python scripts/build_splits.py

Freeze the result: commit a copy to configs/splits.json once the full datasets are in
place, and never change it after the first training run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pianoscribe.data import splits as S


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/splits.json"))
    args = ap.parse_args()

    result: dict = {}

    maestro_csv = args.raw / "maestro-v3.0.0" / "maestro-v3.0.0.csv"
    if not maestro_csv.exists():
        raise SystemExit(f"Missing {maestro_csv}. Run scripts/download_maestro.py first.")
    result["maestro"] = S.maestro_splits(maestro_csv)
    print("maestro:", {k: len(v) for k, v in result["maestro"].items()})

    asap_csv = args.raw / "asap-dataset" / "metadata.csv"
    if asap_csv.exists():
        asap, excluded = S.asap_beat_splits(asap_csv, result["maestro"])
        result["asap_beats"] = asap
        result["asap_excluded_pieces"] = excluded
        print(
            "asap_beats:", {k: len(v) for k, v in asap.items()}, "excluded pieces:", len(excluded)
        )
    else:
        print("ASAP not found, skipped")

    pop_dir = args.raw / "POP909-Dataset" / "POP909"
    if pop_dir.exists():
        songs = [p.name for p in pop_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        result["pop909"] = S.song_level_split(songs)
        print("pop909:", {k: len(v) for k, v in result["pop909"].items()})
    else:
        print("POP909 not found, skipped")

    for name in ("noise", "irs"):
        d = args.raw / name
        if d.exists():
            result[name] = S.split_directory(d)
            print(f"{name}:", {k: len(v) for k, v in result[name].items()})

    S.write_splits(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
