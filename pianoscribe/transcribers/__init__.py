"""Common interface for every transcription model (ScribeNet and the reference models).

A transcriber takes 16 kHz mono float32 audio and returns a Performance
(notes + sustain-pedal intervals). Models are created by name so scripts can say
`--models reference scribenet`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pianoscribe.data.notes import Performance


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> Performance: ...


def get_transcriber(name: str, device: str = "cpu") -> Transcriber:
    if name == "reference":
        from pianoscribe.transcribers.reference import ReferenceTranscriber

        return ReferenceTranscriber(device=device)
    raise ValueError(f"Unknown transcriber '{name}'. Available: reference (scribenet in Phase 2)")


AVAILABLE = ("reference",)
