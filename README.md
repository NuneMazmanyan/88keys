# 88keys

Piano audio → sheet music, with automatic chords and richer arrangements.
Diploma project. Python package name: `pianoscribe`.

| Component | What it does | Status |
|---|---|---|
| ScribeNet | Robust, beat-aware piano transcription model | Phase 2 |
| Score engine | Notes + beats → two-staff MusicXML | Phase 3 |
| HarmoNet + Arranger | Melody → chords → richer piano arrangement | Phase 4 |
| API + web app | Upload, view, play, enrich | Phase 5 |
| **Data pipeline** | Downloads, preprocessing, leakage-safe splits | **Phase 0 (this)** |

## Setup (macOS or Linux)

```bash
brew install uv ffmpeg                # macOS; on Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev                   # creates .venv with Python 3.11
uv sync --extra dev --extra train     # on the GPU machine (adds PyTorch etc.)
```

## Data

Datasets live in `data/` (git-ignored). On a laptop use a small MAESTRO subset; on the GPU
machine download everything.

```bash
# laptop: 30 pieces fetched straight from the remote zip (~2-3 GB instead of ~100 GB)
uv run python scripts/download_maestro.py --subset 20 5 5

# GPU machine: everything
uv run python scripts/download_maestro.py --full

# ASAP, POP909, Chordonomicon
uv sync --extra symbolic && bash scripts/download_symbolic.sh

# 16 kHz mono FLAC + note/pedal labels + manifest.csv
uv run python scripts/preprocess_maestro.py --workers 8

# leakage-safe splits for all datasets -> data/processed/splits.json
uv run python scripts/build_splits.py
```

MAPS has to be requested from Télécom Paris and placed in `data/raw/maps`.

### Split rules

1. MAESTRO keeps its official split.
2. ASAP performances recorded in MAESTRO inherit MAESTRO's split (beat labels).
3. ASAP pieces with any performance in MAESTRO test are excluded from beat training.
4. POP909: seeded song-level 80/10/10.
5. Noise and room impulse responses: stable hash split 80/20 (train / MAESTRO-Wild test).

## Tests

```bash
uv run pytest
```

## Layout

```
configs/        paths and (later) Hydra configs
pianoscribe/    the Python package (data/, models/, score/, enrich/, eval/ ...)
scripts/        download, preprocess, split builders
tests/          unit tests
data/           datasets (git-ignored)
```
