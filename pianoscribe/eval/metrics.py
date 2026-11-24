"""Transcription metrics, following the standard used in piano transcription papers.

Note-level (mir_eval): onset tolerance 50 ms; offset tolerance max(50 ms, 20% of note length);
velocity tolerance 0.1 after mir_eval's velocity rescaling.
Frame-level: piano rolls at 100 frames/s, element-wise precision / recall / F1.

Reference offsets are pedal-extended by default (the convention of the MAESTRO papers):
a key held down by the sustain pedal keeps sounding, so a transcriber hears it as one long
note. Use `ref_offsets="raw"` to evaluate against key releases instead.
"""

from __future__ import annotations

import numpy as np
from mir_eval import transcription, transcription_velocity
from mir_eval.util import midi_to_hz

from pianoscribe.data.notes import Performance, extend_offsets_by_pedal

FRAME_RATE = 100
METRIC_KEYS = (
    "frame_p", "frame_r", "frame_f1",
    "note_p", "note_r", "note_f1",
    "note_off_p", "note_off_r", "note_off_f1",
    "note_off_vel_p", "note_off_vel_r", "note_off_vel_f1",
)  # fmt: skip


def _intervals_pitches(notes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    notes = notes[notes[:, 1] > notes[:, 0]]
    return notes[:, :2], midi_to_hz(notes[:, 2]), notes[:, 3]


def piano_roll(notes: np.ndarray, n_frames: int, frame_rate: int = FRAME_RATE) -> np.ndarray:
    roll = np.zeros((n_frames, 88), dtype=bool)
    for on, off, pitch, _ in notes:
        a = int(round(on * frame_rate))
        b = max(a + 1, int(round(off * frame_rate)))
        p = int(pitch) - 21
        if 0 <= p < 88 and a < n_frames:
            roll[a : min(b, n_frames), p] = True
    return roll


def _prf(tp: float, n_est: float, n_ref: float) -> tuple[float, float, float]:
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(ref: Performance, est: Performance, ref_offsets: str = "pedal") -> dict[str, float]:
    ref_notes = extend_offsets_by_pedal(ref) if ref_offsets == "pedal" else ref.notes
    est_notes = est.notes
    out: dict[str, float] = {}

    n_frames = int(np.ceil(max(ref.duration, est.duration) * FRAME_RATE)) + 1
    r_roll, e_roll = piano_roll(ref_notes, n_frames), piano_roll(est_notes, n_frames)
    tp = float(np.logical_and(r_roll, e_roll).sum())
    out["frame_p"], out["frame_r"], out["frame_f1"] = _prf(tp, e_roll.sum(), r_roll.sum())

    if len(ref_notes) == 0 or len(est_notes) == 0:
        for k in METRIC_KEYS[3:]:
            out[k] = 0.0
        return out

    ri, rp, rv = _intervals_pitches(ref_notes)
    ei, ep, ev = _intervals_pitches(est_notes)

    p, r, f, _ = transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    out["note_p"], out["note_r"], out["note_f1"] = p, r, f

    p, r, f, _ = transcription.precision_recall_f1_overlap(ri, rp, ei, ep)
    out["note_off_p"], out["note_off_r"], out["note_off_f1"] = p, r, f

    p, r, f, _ = transcription_velocity.precision_recall_f1_overlap(ri, rp, rv, ei, ep, ev)
    out["note_off_vel_p"], out["note_off_vel_r"], out["note_off_vel_f1"] = p, r, f
    return out
