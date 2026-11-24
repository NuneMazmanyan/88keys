import numpy as np
import pytest

from pianoscribe.data.notes import Performance
from pianoscribe.eval.metrics import evaluate

REF = Performance(
    notes=np.array(
        [[0.0, 0.5, 60, 80], [0.0, 0.5, 64, 70], [1.0, 1.5, 67, 90], [2.0, 2.4, 72, 60]],
        dtype=float,
    ),
    pedal=np.zeros((0, 2)),
)


def test_perfect_transcription_scores_one():
    m = evaluate(REF, REF)
    for k in ("frame_f1", "note_f1", "note_off_f1", "note_off_vel_f1"):
        assert m[k] == pytest.approx(1.0)


def test_onset_tolerance_and_missing_note():
    est = REF.notes.copy()
    est[0, 0] += 0.03  # within 50 ms -> still correct
    est[2, 0] += 0.08  # outside 50 ms -> wrong
    est = est[:3]  # last note missing
    m = evaluate(REF, Performance(notes=est, pedal=np.zeros((0, 2))))
    assert m["note_p"] == pytest.approx(2 / 3)
    assert m["note_r"] == pytest.approx(2 / 4)


def test_wrong_offset_hurts_only_offset_metric():
    est = REF.notes.copy()
    est[:, 1] += 0.5
    m = evaluate(REF, Performance(notes=est, pedal=np.zeros((0, 2))))
    assert m["note_f1"] == pytest.approx(1.0)
    assert m["note_off_f1"] < 0.5


def test_pedal_extends_reference_offsets():
    ref = Performance(notes=REF.notes[:1].copy(), pedal=np.array([[0.2, 1.0]]))
    est = Performance(notes=np.array([[0.0, 1.0, 60, 80]]), pedal=np.zeros((0, 2)))
    assert evaluate(ref, est)["note_off_f1"] == pytest.approx(1.0)
    assert evaluate(ref, est, ref_offsets="raw")["note_off_f1"] == pytest.approx(0.0)


def test_empty_estimate():
    m = evaluate(REF, Performance(notes=np.zeros((0, 4)), pedal=np.zeros((0, 2))))
    assert m["note_f1"] == 0.0 and m["frame_f1"] == 0.0
