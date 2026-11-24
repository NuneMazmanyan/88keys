"""Download the OpenSLR RIRS_NOISES collection (SLR28, ~1.3 GB, Apache 2.0).

Real room impulse responses and real noises used for MAESTRO-Wild (test files) and,
in Phase 2, for training augmentation (train files).

  python scripts/download_rirs_noises.py
"""

from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path

URL = "https://www.openslr.org/resources/28/rirs_noises.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/raw/rirs_noises"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "RIRS_NOISES").exists():
        print("Already present")
        return
    zip_path = args.out / "rirs_noises.zip"
    subprocess.run(["curl", "-L", "-C", "-", "-o", str(zip_path), URL], check=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.out)
    zip_path.unlink()
    print(f"Extracted to {args.out / 'RIRS_NOISES'}")


if __name__ == "__main__":
    main()
