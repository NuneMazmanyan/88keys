"""Note and pedal representation shared by every model, dataset and evaluator.

A performance is stored as two float arrays:
  notes: (N, 4) columns = onset_sec, offset_sec, midi_pitch, velocity   (sorted by onset, pitch)
  pedal: (M, 2) columns = start_sec, end_sec of sustain-pedal-down intervals
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

PIANO_MIN_PITCH = 21  # A0
PIANO_MAX_PITCH = 108  # C8
SUSTAIN_CC = 64
PEDAL_THRESHOLD = 64  # MIDI convention: value >= 64 means pedal down


@dataclass
class Performance:
    notes: np.ndarray  # (N, 4) float64
    pedal: np.ndarray  # (M, 2) float64

    @property
    def duration(self) -> float:
        end = 0.0
        if len(self.notes):
            end = max(end, float(self.notes[:, 1].max()))
        if len(self.pedal):
            end = max(end, float(self.pedal[:, 1].max()))
        return end

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, notes=self.notes, pedal=self.pedal)

    @classmethod
    def load(cls, path: str | Path) -> Performance:
        with np.load(path) as f:
            return cls(notes=f["notes"], pedal=f["pedal"])


def _pedal_intervals(control_changes: list, end_time: float) -> np.ndarray:
    events = sorted(
        (cc.time, cc.value >= PEDAL_THRESHOLD) for cc in control_changes if cc.number == SUSTAIN_CC
    )
    intervals: list[tuple[float, float]] = []
    down_at: float | None = None
    for t, is_down in events:
        if is_down and down_at is None:
            down_at = t
        elif not is_down and down_at is not None:
            if t > down_at:
                intervals.append((down_at, t))
            down_at = None
    if down_at is not None and end_time > down_at:
        intervals.append((down_at, end_time))
    return np.asarray(intervals, dtype=np.float64).reshape(-1, 2)


def performance_from_midi(path: str | Path) -> Performance:
    """Parse a performance MIDI (e.g. MAESTRO) into notes and sustain-pedal intervals.

    Notes outside the 88-key range and drum tracks are dropped. Note offsets are the raw
    key releases; pedal-extended offsets are computed separately when needed.
    """
    pm = pretty_midi.PrettyMIDI(str(path))
    rows = []
    ccs = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        ccs.extend(inst.control_changes)
        for n in inst.notes:
            if PIANO_MIN_PITCH <= n.pitch <= PIANO_MAX_PITCH and n.end > n.start:
                rows.append((n.start, n.end, n.pitch, n.velocity))
    notes = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if len(notes):
        notes = notes[np.lexsort((notes[:, 2], notes[:, 0]))]
    pedal = _pedal_intervals(ccs, pm.get_end_time())
    return Performance(notes=notes, pedal=pedal)


def crop(perf: Performance, end_sec: float) -> Performance:
    """Keep notes that start before `end_sec`; clip offsets and pedal intervals to it."""
    notes = perf.notes[perf.notes[:, 0] < end_sec].copy()
    notes[:, 1] = np.minimum(notes[:, 1], end_sec)
    pedal = perf.pedal[perf.pedal[:, 0] < end_sec].copy()
    pedal[:, 1] = np.minimum(pedal[:, 1], end_sec)
    return Performance(notes=notes, pedal=pedal)


def extend_offsets_by_pedal(perf: Performance) -> np.ndarray:
    """Return a copy of notes whose offsets are extended to the end of the sustain pedal.

    A sustained note still ends when the same pitch is struck again.
    """
    notes = perf.notes.copy()
    if not len(notes) or not len(perf.pedal):
        return notes
    for i, (on, off, pitch, _) in enumerate(notes):
        for p_start, p_end in perf.pedal:
            if p_start <= off < p_end:
                notes[i, 1] = p_end
                break
        # cut at next onset of the same pitch
        later = notes[(notes[:, 2] == pitch) & (notes[:, 0] > on), 0]
        if len(later):
            notes[i, 1] = min(notes[i, 1], later.min())
    return notes
