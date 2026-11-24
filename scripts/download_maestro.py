"""Download MAESTRO v3.0.0.

Two modes:
  --full            the whole ~100 GB zip (use this on the GPU machine)
  --subset N_TRAIN N_VAL N_TEST
                    only a few pieces, fetched straight out of the remote zip with HTTP range
                    requests (no 100 GB download). Use this on a laptop for development.

Examples:
  python scripts/download_maestro.py --subset 20 5 5
  python scripts/download_maestro.py --full
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip
from tqdm import tqdm

URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
ZIP_ROOT = "maestro-v3.0.0"


def download_full(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "maestro-v3.0.0.zip"
    if not zip_path.exists():
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(zip_path), URL], check=True)
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"Done. You can now delete {zip_path} to free ~100 GB.")


def pick_subset(
    df: pd.DataFrame, counts: dict[str, int], max_sec: float, seed: int
) -> pd.DataFrame:
    rng = random.Random(seed)
    picked = []
    for split, n in counts.items():
        pool = df[(df["split"] == split) & (df["duration"] <= max_sec)].index.tolist()
        rng.shuffle(pool)
        picked.extend(pool[:n])
    return df.loc[sorted(picked)]


def download_subset(out_dir: Path, counts: dict[str, int], max_sec: float, seed: int) -> None:
    root = out_dir / ZIP_ROOT
    root.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as rz:
        csv_bytes = rz.read(f"{ZIP_ROOT}/maestro-v3.0.0.csv")
        (root / "maestro-v3.0.0.csv").write_bytes(csv_bytes)
        df = pd.read_csv(io.BytesIO(csv_bytes))
        sub = pick_subset(df, counts, max_sec, seed)
        files = [f for _, r in sub.iterrows() for f in (r["midi_filename"], r["audio_filename"])]
        for rel in tqdm(files, desc="MAESTRO subset"):
            dst = root / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with rz.open(f"{ZIP_ROOT}/{rel}") as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
    sub.to_csv(root / "subset.csv", index=False)
    print(f"Saved {len(sub)} performances to {root} (list in subset.csv)")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--full", action="store_true")
    g.add_argument("--subset", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--max-sec", type=float, default=400.0, help="subset: skip longer pieces")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if args.full:
        download_full(args.out)
    else:
        t, v, te = args.subset
        download_subset(
            args.out, {"train": t, "validation": v, "test": te}, args.max_sec, args.seed
        )


if __name__ == "__main__":
    main()
