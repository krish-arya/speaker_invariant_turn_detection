"""Synthetic speech-like audio with controllable speaker identity and phrase-final prosody.

This is NOT speech synthesis. It produces harmonic, syllable-structured signals so that the
real pipeline (VAD → detector → policy → orchestrator) can be exercised repeatably without
recordings, and so learned detectors can be smoke-tested. Two families of cues are separated
on purpose:

* **speaker identity** — f0 level, formant scale, breathiness, loudness, speaking rate;
* **turn-final prosody** — falling f0 and final-syllable lengthening at turn ends versus a
  level/rising contour before a within-turn pause.

Results obtained on synthetic voices are pipeline checks, never evidence about real speakers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Contour = Literal["final", "hold", "question", "backchannel"]

# Rough vowel formant centres (Hz) for an adult voice; scaled per speaker.
_VOWELS = np.array(
    [
        [730, 1090, 2440],  # a
        [530, 1840, 2480],  # e
        [270, 2290, 3010],  # i
        [570, 840, 2410],  # o
        [300, 870, 2240],  # u
        [640, 1190, 2390],  # schwa-ish
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    voice_id: int
    f0_hz: float  # mean fundamental frequency
    f0_jitter_st: float  # per-syllable pitch variability (semitones)
    formant_scale: float  # vocal-tract length proxy
    rate: float  # speaking-rate multiplier (1.0 ≈ 5 syllables/s)
    breathiness: float  # 0..1 aspiration-noise mix
    level_dbfs: float  # target speech RMS level
    final_fall_st: float  # f0 fall over the last syllables of a turn (semitones)
    hold_rise_st: float  # f0 movement before a within-turn pause (semitones)

    @classmethod
    def from_id(cls, voice_id: int) -> VoiceProfile:
        """Deterministic voice parameters from an integer id (same id → same voice)."""
        rng = np.random.default_rng(10_007 * (voice_id + 1))
        low = rng.random() < 0.5
        f0 = float(rng.uniform(85, 150) if low else rng.uniform(165, 265))
        return cls(
            voice_id=voice_id,
            f0_hz=f0,
            f0_jitter_st=float(rng.uniform(0.6, 2.0)),
            formant_scale=float(rng.uniform(0.85, 1.0) if low else rng.uniform(1.0, 1.2)),
            rate=float(rng.uniform(0.8, 1.25)),
            breathiness=float(rng.uniform(0.02, 0.25)),
            level_dbfs=float(rng.uniform(-28, -18)),
            final_fall_st=float(rng.uniform(3.0, 7.0)),
            hold_rise_st=float(rng.uniform(-0.5, 1.5)),
        )


def _syllable(
    voice: VoiceProfile,
    f0_start: float,
    f0_end: float,
    duration_s: float,
    vowel: np.ndarray,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = max(int(duration_s * sr), 16)
    f0 = np.linspace(f0_start, f0_end, n)
    f0 *= 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * np.arange(n) / sr)  # mild vibrato
    phase = 2 * np.pi * np.cumsum(f0) / sr
    n_harm = int(min(40, 3800 // max(f0.max(), 60)))
    k = np.arange(1, n_harm + 1)[:, None]
    harm_freq = k * f0[None, :].mean()
    formants = vowel * voice.formant_scale
    gain = np.zeros_like(harm_freq)
    for fc, bw, amp in zip(formants, (90.0, 120.0, 180.0), (1.0, 0.6, 0.3), strict=True):
        gain += amp * np.exp(-0.5 * ((harm_freq - fc) / bw) ** 2)
    gain += 0.05 / k  # spectral tilt floor
    signal = (gain * np.sin(k * phase[None, :])).sum(axis=0)
    noise = rng.standard_normal(n) * voice.breathiness * np.std(signal)
    signal = signal + noise
    # attack / release envelope; slight dip between syllables keeps VAD continuous
    env = np.ones(n)
    a, r = min(int(0.02 * sr), n // 3), min(int(0.035 * sr), n // 3)
    env[:a] = np.linspace(0.25, 1.0, a)
    env[n - r :] = np.linspace(1.0, 0.25, r)
    return (signal * env).astype(np.float64)


def synth_phrase(
    voice: VoiceProfile,
    n_syllables: int,
    contour: Contour,
    rng: np.random.Generator,
    sr: int = 16_000,
) -> np.ndarray:
    """One continuous stretch of speech (no internal pauses long enough to end a turn)."""
    n_syllables = max(1, n_syllables)
    base_dur = 0.19 / voice.rate
    out: list[np.ndarray] = []
    # declination across the phrase, plus the contour on the last syllables
    decl = np.linspace(0.0, -1.5, n_syllables)
    for i in range(n_syllables):
        jitter = rng.normal(0.0, voice.f0_jitter_st)
        st_start = decl[i] + jitter
        st_end = st_start
        dur = base_dur * rng.uniform(0.75, 1.3)
        last = i == n_syllables - 1
        second_last = i == n_syllables - 2
        if contour == "final":
            if second_last:
                st_end = st_start - 0.35 * voice.final_fall_st
            if last:
                st_start = decl[i] - 0.35 * voice.final_fall_st
                st_end = st_start - 0.65 * voice.final_fall_st
                dur *= 1.45  # phrase-final lengthening
        elif contour == "hold" and last:
            st_end = st_start + voice.hold_rise_st
        elif contour == "question" and last:
            st_end = st_start + 4.0
            dur *= 1.2
        elif contour == "backchannel":
            st_end = st_start - 1.0
            dur *= 1.2
        f0a = voice.f0_hz * 2 ** (st_start / 12)
        f0b = voice.f0_hz * 2 ** (st_end / 12)
        vowel = _VOWELS[rng.integers(len(_VOWELS))]
        out.append(_syllable(voice, f0a, f0b, dur, vowel, sr, rng))
        if not last and rng.random() < 0.3:  # short intra-phrase gap (< VAD hangover)
            out.append(np.zeros(int(rng.uniform(0.01, 0.03) * sr)))
    speech = np.concatenate(out)
    rms = np.sqrt(np.mean(speech**2)) + 1e-9
    speech *= 10 ** (voice.level_dbfs / 20) / rms
    return speech.astype(np.float32)


def noise_bed(n_samples: int, level_dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """Pink-ish background noise at a given RMS level."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.arange(spectrum.size, dtype=np.float64)
    spectrum /= np.sqrt(np.maximum(freqs, 1.0))  # 1/f power tilt
    pink = np.fft.irfft(spectrum, n=n_samples)
    mix = 0.6 * pink / (np.std(pink) + 1e-9) + 0.4 * white
    mix *= 10 ** (level_dbfs / 20) / (np.sqrt(np.mean(mix**2)) + 1e-9)
    return mix.astype(np.float32)
