#!/usr/bin/env bash
# Download the small datasets: ASAP (beats + scores), POP909 (melody + chords),
# Chordonomicon (chord progressions). Safe to re-run.
set -euo pipefail

RAW="${DATA_ROOT:-data}/raw"
mkdir -p "$RAW"

clone() {  # clone <url> <dir>
  if [ -d "$RAW/$2/.git" ]; then
    echo "$2 already present"
  else
    git clone --depth 1 "$1" "$RAW/$2"
  fi
}

clone https://github.com/fosfrancesco/asap-dataset.git asap-dataset
clone https://github.com/music-x-lab/POP909-Dataset.git POP909-Dataset

# Chordonomicon from Hugging Face (needs: uv sync --extra symbolic)
python - <<'PY'
import os
from pathlib import Path
out = Path(os.environ.get("DATA_ROOT", "data")) / "raw" / "chordonomicon"
if (out / "chordonomicon.parquet").exists():
    print("chordonomicon already present")
else:
    from datasets import load_dataset
    ds = load_dataset("ailsntua/Chordonomicon", split="train")
    out.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(out / "chordonomicon.parquet"))
    print(f"chordonomicon: {len(ds)} songs")
PY

echo
echo "MAPS must be requested manually from Telecom Paris (ADASP); place it in $RAW/maps."
echo "Noise (MUSAN) and impulse responses go in $RAW/noise and $RAW/irs (Phase 2)."
