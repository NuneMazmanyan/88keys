"""MAESTRO-Wild: reproducible real-world recording conditions for testing robustness.

Sources: the OpenSLR "RIRS_NOISES" collection (SLR28, Apache 2.0): real room impulse
responses (RWCP, REVERB challenge, AIR) and real background noises. Files are assigned to
train / test with a stable hash; MAESTRO-Wild uses only the *test* files, so a model trained
with the train files never hears these rooms or noises.

Every degradation keeps the piano's timing unchanged (reverb starts at the impulse
response's direct sound; codec delay is removed by cross-correlation), so the original
note labels stay valid.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy import signal

from pianoscribe.data.audio import SAMPLE_RATE, load_audio
from pianoscribe.data.splits import hash_split

# ---------------------------------------------------------------- conditions


@dataclass(frozen=True)
class Condition:
    name: str
    reverb: str | None = None  # "medium" (RT60 0.3-0.8 s) or "long" (> 0.8 s)
    snr_db: float | None = None
    phone: bool = False
    description: str = field(default="", compare=False)


CONDITIONS: tuple[Condition, ...] = (
    Condition("clean", description="Original MAESTRO audio"),
    Condition("reverb_medium", reverb="medium", description="Room, RT60 0.3-0.8 s"),
    Condition("reverb_long", reverb="long", description="Large room, RT60 > 0.8 s"),
    Condition("noise_20", snr_db=20, description="Background noise, SNR 20 dB"),
    Condition("noise_10", snr_db=10, description="Background noise, SNR 10 dB"),
    Condition("noise_5", snr_db=5, description="Background noise, SNR 5 dB"),
    Condition("phone", phone=True, description="Phone mic band-limit + AAC 48 kbps"),
    Condition(
        "combined", reverb="medium", snr_db=10, phone=True,
        description="Room + noise 10 dB + phone",
    ),
)  # fmt: skip
CONDITION_BY_NAME = {c.name: c for c in CONDITIONS}


def seed_for(*parts: str) -> int:
    return int(hashlib.sha1("|".join(parts).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- catalog of IRs and noises


def catalog(rirs_noises_root: str | Path) -> dict[str, dict[str, list[str]]]:
    """List real RIRs and noises under RIRS_NOISES, split train/test by stable hash."""
    root = Path(rirs_noises_root)
    real = root / "real_rirs_isotropic_noises"
    point = root / "pointsource_noises"
    irs, noises = [], []
    for p in sorted(real.glob("*.wav")):
        (noises if "noise" in p.name.lower() else irs).append(str(p.relative_to(root)))
    for p in sorted(point.glob("*.wav")):
        noises.append(str(p.relative_to(root)))
    out: dict[str, dict[str, list[str]]] = {}
    for kind, files in (("irs", irs), ("noises", noises)):
        out[kind] = {"train": [], "test": []}
        for f in files:
            out[kind][hash_split(f)].append(f)
    return out


def load_mono(path: str | Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = data[:, 0]  # first channel of multichannel IRs / noises
    if sr != SAMPLE_RATE:
        x = soxr.resample(x, sr, SAMPLE_RATE)
    return x.astype(np.float32)


def rt60(ir: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Reverberation time from the Schroeder decay curve (T20 extrapolated to 60 dB)."""
    energy = ir.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-12)
    try:
        i5 = np.where(edc_db <= -5)[0][0]
        i25 = np.where(edc_db <= -25)[0][0]
    except IndexError:
        return float("nan")
    return float(3 * (i25 - i5) / sr)


def ir_bank(root: str | Path, files: list[str]) -> dict[str, list[np.ndarray]]:
    """Load IRs, align each to its direct-sound peak, and group by reverb length."""
    bank: dict[str, list[np.ndarray]] = {"medium": [], "long": []}
    for f in files:
        ir = load_mono(Path(root) / f)
        ir = ir[int(np.argmax(np.abs(ir))) :]
        t = rt60(ir)
        if 0.3 <= t <= 0.8:
            bank["medium"].append(ir)
        elif t > 0.8:
            bank["long"].append(ir)
    return bank


# ---------------------------------------------------------------- degradations


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def add_reverb(x: np.ndarray, ir: np.ndarray) -> np.ndarray:
    ir = ir / (np.abs(ir).max() + 1e-12)
    y = signal.fftconvolve(x, ir)[: len(x)]
    return (y * rms(x) / rms(y)).astype(np.float32)


def add_noise(x: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator):
    if len(noise) < len(x):
        noise = np.tile(noise, int(np.ceil(len(x) / len(noise))))
    start = int(rng.integers(0, len(noise) - len(x) + 1))
    n = noise[start : start + len(x)]
    n = n * (rms(x) / rms(n)) / (10 ** (snr_db / 20))
    return (x + n).astype(np.float32)


def _align(reference: np.ndarray, y: np.ndarray, max_lag: int = SAMPLE_RATE // 5) -> np.ndarray:
    """Shift `y` so it lines up with `reference` (removes codec priming delay)."""
    n = min(len(reference), len(y), SAMPLE_RATE * 20)
    corr = signal.correlate(y[: n + max_lag], reference[:n], mode="valid", method="fft")
    lag = int(np.argmax(np.abs(corr[: max_lag + 1])))
    y = y[lag:]
    if len(y) < len(reference):
        y = np.pad(y, (0, len(reference) - len(y)))
    return y[: len(reference)]


def phone_simulation(x: np.ndarray, bitrate: str = "48k") -> np.ndarray:
    """Band-limit like a phone microphone, add mild compression, then AAC encode/decode."""
    sos = signal.butter(4, [150, 7000], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    y = signal.sosfilt(sos, x)
    peak = np.abs(y).max() + 1e-12
    y = np.tanh(2.0 * y / peak) / np.tanh(2.0)  # soft compression
    y = (y * rms(x) / rms(y)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        wav, m4a, back = Path(td, "a.wav"), Path(td, "a.m4a"), Path(td, "b.wav")
        sf.write(wav, np.clip(y, -1, 1), SAMPLE_RATE)
        run = dict(check=True, capture_output=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", bitrate, str(m4a)], **run
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(m4a), "-ar", str(SAMPLE_RATE), "-ac", "1", str(back)], **run
        )  # noqa: E501
        z = load_audio(back)
    return _align(y, z)


def degrade(
    x: np.ndarray,
    cond: Condition,
    piece_id: str,
    irs: dict[str, list[np.ndarray]] | None,
    noises: list[np.ndarray] | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed_for(piece_id, cond.name))
    y = x
    if cond.reverb:
        if not irs or not irs.get(cond.reverb):
            raise RuntimeError(f"No '{cond.reverb}' impulse responses available")
        bank = irs[cond.reverb]
        y = add_reverb(y, bank[int(rng.integers(len(bank)))])
    if cond.snr_db is not None:
        if not noises:
            raise RuntimeError("No noise files available")
        y = add_noise(y, noises[int(rng.integers(len(noises)))], cond.snr_db, rng)
    if cond.phone:
        y = phone_simulation(y)
    peak = np.abs(y).max()
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)
