# 88keys

Piano audio → sheet music, with automatic chords and richer arrangements.
Diploma project. Python package name: `pianoscribe`.

| Component | What it does | Status |
|---|---|---|
| ScribeNet | Robust, beat-aware piano transcription model | Phase 2 |
| Score engine | Notes + beats → two-staff MusicXML | Phase 3 |
| HarmoNet + Arranger | Melody → chords → richer piano arrangement | Phase 4 |
| API + web app | Upload, view, play, enrich | Phase 5 |
| Data pipeline | Downloads, preprocessing, leakage-safe splits | Phase 0 ✓ |
| **Evaluation + MAESTRO-Wild** | Metrics, real-world test conditions, benchmark runner | **Phase 1** |

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

## Evaluation and MAESTRO-Wild

MAESTRO-Wild re-records the MAESTRO test pieces under real-world conditions, using real
room impulse responses and real noises (OpenSLR RIRS_NOISES, test half only):

| Condition | What it simulates |
|---|---|
| clean | original audio |
| reverb_medium / reverb_long | room, RT60 0.3–0.8 s / > 0.8 s |
| noise_20 / noise_10 / noise_5 | background noise at that SNR (dB) |
| phone | phone mic band-limit + soft compression + AAC 48 kbps |
| combined | room + noise 10 dB + phone |

Timing is preserved (reverb aligned to the direct sound, codec delay removed), so the
original note labels stay exact.

```bash
uv sync --extra dev --extra reference                    # adds the reference model
uv run python scripts/download_rirs_noises.py            # ~1.3 GB
uv run python scripts/build_wild.py --max-seconds 60     # quick version; omit for full pieces
uv run python scripts/run_benchmark.py --models oracle reference
```

Results go to `outputs/benchmark/results.csv` (every piece, every metric) and
`outputs/benchmark/summary.md`. `oracle` returns the ground truth and must score 100;
it checks the harness itself.

Metrics follow the field's standard (mir_eval): note F1 with 50 ms onset tolerance, note F1
with offsets, with offsets + velocity, and frame F1. Reference offsets are extended by the
sustain pedal, as in the MAESTRO papers.

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
