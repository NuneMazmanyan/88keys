"""Audio loading and conversion to the project's canonical format (16 kHz, mono, FLAC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any file soundfile can read as float32 mono at `sr`.

    For mp3/m4a/webm uploads, convert with ffmpeg first (see `ffmpeg_to_wav`).
    """
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if file_sr != sr:
        mono = soxr.resample(mono, file_sr, sr, quality="HQ")
    return mono.astype(np.float32)


def save_flac(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio, sr, format="FLAC", subtype="PCM_16")


def convert_to_canonical(src: str | Path, dst: str | Path, sr: int = SAMPLE_RATE) -> float:
    """Resample `src` to mono FLAC at `sr`; returns the duration in seconds."""
    audio = load_audio(src, sr)
    save_flac(dst, audio, sr)
    return len(audio) / sr
