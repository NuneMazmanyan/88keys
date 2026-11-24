import numpy as np
import pretty_midi

from pianoscribe.data.notes import Performance, extend_offsets_by_pedal, performance_from_midi


def _make_midi(path):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    piano.notes += [
        pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5),
        pretty_midi.Note(velocity=70, pitch=64, start=0.0, end=0.4),
        pretty_midi.Note(velocity=60, pitch=60, start=1.5, end=2.0),
        pretty_midi.Note(velocity=50, pitch=10, start=0.1, end=0.2),  # below A0, dropped
    ]
    piano.control_changes += [
        pretty_midi.ControlChange(number=64, value=127, time=0.2),
        pretty_midi.ControlChange(number=64, value=0, time=1.0),
        pretty_midi.ControlChange(number=64, value=100, time=1.8),  # never released
    ]
    pm.instruments.append(piano)
    pm.write(str(path))


def test_parse_notes_and_pedal(tmp_path):
    midi = tmp_path / "x.mid"
    _make_midi(midi)
    perf = performance_from_midi(midi)
    assert perf.notes.shape == (3, 4)
    assert list(perf.notes[:, 2]) == [60, 64, 60]  # sorted by onset, then pitch
    assert np.allclose(perf.pedal[0], [0.2, 1.0], atol=1e-3)
    assert perf.pedal.shape == (2, 2)  # open pedal closed at end of file


def test_save_load_roundtrip(tmp_path):
    midi = tmp_path / "x.mid"
    _make_midi(midi)
    perf = performance_from_midi(midi)
    perf.save(tmp_path / "x.npz")
    back = Performance.load(tmp_path / "x.npz")
    assert np.array_equal(back.notes, perf.notes)
    assert np.array_equal(back.pedal, perf.pedal)


def test_pedal_extension_stops_at_restrike(tmp_path):
    midi = tmp_path / "x.mid"
    _make_midi(midi)
    ext = extend_offsets_by_pedal(performance_from_midi(midi))
    assert np.isclose(ext[0, 1], 1.0, atol=1e-3)  # C4 sustained until pedal up
    assert np.isclose(ext[1, 1], 1.0, atol=1e-3)
