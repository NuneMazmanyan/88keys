import numpy as np
import pytest

from pianoscribe.data.audio import SAMPLE_RATE
from pianoscribe.eval import wild


def _clicks(seconds=3.0):
    x = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    x[SAMPLE_RATE // 2 :: SAMPLE_RATE] = 1.0  # one click per second, from 0.5 s
    return x


def _synthetic_ir(rt=0.6, delay=200):
    rng = np.random.default_rng(0)
    n = int(1.5 * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt)
    ir = np.concatenate([np.zeros(delay), [1.0], 0.3 * tail]).astype(np.float32)
    return ir


def test_rt60_estimate():
    assert wild.rt60(_synthetic_ir(0.6)[200:]) == pytest.approx(0.6, rel=0.2)


def test_reverb_keeps_timing_after_direct_sound_alignment():
    ir = _synthetic_ir()
    ir = ir[int(np.argmax(np.abs(ir))) :]
    y = wild.add_reverb(_clicks(), ir)
    first = int(np.argmax(np.abs(y) > 0.5 * np.abs(y).max()))
    assert abs(first - SAMPLE_RATE // 2) <= 2


def test_noise_snr():
    x = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE).astype(np.float32)
    noise = np.random.default_rng(1).standard_normal(SAMPLE_RATE).astype(np.float32)
    y = wild.add_noise(x, noise, 10, np.random.default_rng(2))
    snr = 20 * np.log10(wild.rms(x) / wild.rms(y - x))
    assert snr == pytest.approx(10, abs=0.1)


def test_phone_simulation_keeps_alignment():
    x = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)
    t = np.arange(int(0.05 * SAMPLE_RATE)) / SAMPLE_RATE
    burst = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    x[SAMPLE_RATE : SAMPLE_RATE + len(burst)] = burst
    y = wild.phone_simulation(x)
    assert len(y) == len(x)
    env = np.abs(y)
    onset = int(np.argmax(env > 0.3 * env.max()))
    assert abs(onset - SAMPLE_RATE) < 0.01 * SAMPLE_RATE  # within 10 ms


def test_degrade_is_deterministic():
    x = _clicks()
    noise = np.random.default_rng(3).standard_normal(SAMPLE_RATE * 4).astype(np.float32)
    cond = wild.CONDITION_BY_NAME["noise_10"]
    a = wild.degrade(x, cond, "piece", None, [noise])
    b = wild.degrade(x, cond, "piece", None, [noise])
    assert np.array_equal(a, b)
