import numpy as np

from turnsense.audio.frames import FrameBuffer, float_to_pcm16, pcm16_to_float, rms_dbfs
from turnsense.audio.vad import EnergyVAD
from turnsense.sim.voice import VoiceProfile, noise_bed, synth_phrase


def test_frames_have_exact_sample_derived_timestamps():
    fb = FrameBuffer()
    frames = fb.push(np.zeros(16_000 * 3 // 2, dtype=np.float32))  # 1.5 s in one chunk
    assert len(frames) == 75
    assert [f.seq for f in frames[:3]] == [1, 2, 3]
    assert frames[0].start_ms == 0.0 and frames[-1].end_ms == 1500.0
    assert all(f.samples.size == 320 for f in frames)


def test_frames_carry_partial_chunks_across_pushes():
    fb = FrameBuffer()
    out = fb.push(np.zeros(100, dtype=np.float32)) + fb.push(np.zeros(600, dtype=np.float32))
    assert len(out) == 2 and out[1].start_ms == 20.0
    assert fb.clock_ms == 700 / 16


def test_gap_is_recorded_not_fabricated():
    fb = FrameBuffer(gap_tolerance_ms=40)
    fb.push(np.zeros(320, dtype=np.float32), capture_ms=0.0)
    frames = fb.push(np.zeros(320, dtype=np.float32), capture_ms=520.0)  # 500 ms missing
    assert len(fb.gaps) == 1 and abs(fb.gaps[0].missing_ms - 500.0) < 1e-6
    assert frames[0].start_ms == 520.0  # clock jumped; no zero-filled frames were invented
    assert fb.dropped_frames == 25


def test_resampling_and_pcm_roundtrip():
    fb = FrameBuffer()
    frames = fb.push(np.zeros(4800, dtype=np.float32), sample_rate=48_000)  # 100 ms
    assert len(frames) == 5
    x = np.array([0.0, 0.5, -0.5, 0.999], dtype=np.float32)
    assert np.allclose(pcm16_to_float(float_to_pcm16(x)), x, atol=1e-4)
    assert rms_dbfs(np.zeros(10, dtype=np.float32)) < -100


def test_energy_vad_tracks_synthetic_speech_boundaries():
    rng = np.random.default_rng(0)
    voice = VoiceProfile.from_id(5)
    speech = synth_phrase(voice, 6, "final", rng)
    audio = np.concatenate([np.zeros(8000, np.float32), speech, np.zeros(16_000, np.float32)])
    audio += noise_bed(audio.size, -62, rng)
    vad = EnergyVAD()
    changes = [(f.end_ms, r.is_speech) for f in FrameBuffer().push(audio) if (r := vad.process(f)).changed]
    start_ms, end_ms = 500.0, 500.0 + speech.size / 16
    assert changes[0][1] is True and abs(changes[0][0] - start_ms) <= 80
    assert changes[-1][1] is False and abs(changes[-1][0] - end_ms) <= 80
    assert vad.process(FrameBuffer().push(np.zeros(320, np.float32))[0]).probability is None


def test_voice_profiles_are_deterministic_and_distinct():
    assert VoiceProfile.from_id(3) == VoiceProfile.from_id(3)
    assert VoiceProfile.from_id(3).f0_hz != VoiceProfile.from_id(4).f0_hz
