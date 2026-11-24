"""Reference model: high-resolution piano transcription (Kong et al., 2021).

Public pretrained model used only as the point of comparison for ScribeNet.
Install with `uv sync --extra reference`. The checkpoint (~170 MB) downloads on first use.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pianoscribe.data.notes import Performance

CHECKPOINT_URL = (
    "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
DEFAULT_CHECKPOINT = (
    Path(os.environ.get("DATA_ROOT", "data")) / "checkpoints" / "reference_kong2021.pth"
)
MIN_BYTES = 160_000_000  # the package treats smaller files as incomplete downloads


def ensure_checkpoint(path: Path = DEFAULT_CHECKPOINT) -> Path:
    if not path.exists() or path.stat().st_size < MIN_BYTES:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading reference checkpoint (~170 MB) to {path}")
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(path), CHECKPOINT_URL], check=True)
    return path


class ReferenceTranscriber:
    name = "reference"

    def __init__(self, device: str = "cpu", checkpoint: Path = DEFAULT_CHECKPOINT):
        from piano_transcription_inference import PianoTranscription

        ckpt = ensure_checkpoint(checkpoint)
        self._model = PianoTranscription(device=device, checkpoint_path=str(ckpt))

    def transcribe(self, audio: np.ndarray) -> Performance:
        with tempfile.TemporaryDirectory() as td:
            result = self._model.transcribe(audio, str(Path(td) / "out.mid"))
        notes = np.asarray(
            [
                (e["onset_time"], e["offset_time"], e["midi_note"], e["velocity"])
                for e in result["est_note_events"]
            ],
            dtype=np.float64,
        ).reshape(-1, 4)
        pedal = np.asarray(
            [(e["onset_time"], e["offset_time"]) for e in result.get("est_pedal_events", [])],
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(notes):
            notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
        return Performance(notes=notes, pedal=pedal)
