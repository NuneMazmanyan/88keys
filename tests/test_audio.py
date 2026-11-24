import numpy as np
import soundfile as sf

from pianoscribe.data.audio import SAMPLE_RATE, convert_to_canonical, load_audio


def test_stereo_44k_to_mono_16k(tmp_path):
    sr = 44100
    t = np.arange(sr * 2) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(tmp_path / "in.wav", np.stack([tone, tone], axis=1), sr)

    dur = convert_to_canonical(tmp_path / "in.wav", tmp_path / "out.flac")
    assert abs(dur - 2.0) < 0.01

    info = sf.info(tmp_path / "out.flac")
    assert info.samplerate == SAMPLE_RATE and info.channels == 1

    audio = load_audio(tmp_path / "out.flac")
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * SAMPLE_RATE / len(audio)
    assert abs(peak_hz - 440) < 2
