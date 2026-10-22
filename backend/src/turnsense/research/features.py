"""Streaming acoustic features for the learned turn detector (numpy only, no torch needed).

Per analysis window (25 ms Hann, 10 ms hop, 1024-point FFT):

* 40 log-mel band energies (HTK mel scale, 60–7800 Hz);
* periodicity — peak of the window-compensated normalised autocorrelation in the
  70–400 Hz lag range (≈ voicing strength);
* log-f0 — ``log2(f0 / 100 Hz)`` at that peak when voiced (periodicity > 0.45), else 0.

Alignment: the signal is left-padded with ``win - hop`` zeros so that exactly two windows
complete per 20 ms audio frame and the last window of frame *m* ends on that frame's final
sample. Batch (``LogMelPitch.batch``) and streaming (``StreamingFeatures.push``) therefore
produce identical feature sequences — a property the tests check.

Pitch and spectral shape are where speaker identity lives (f0 level, formants) *and* where
turn-final prosody lives (f0 movement, lengthening). Both are deliberately available, so the
anti-transfer objective has something to suppress.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    win_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 1024
    n_mels: int = 40
    fmin: float = 60.0
    fmax: float = 7800.0
    pitch: bool = True
    f0_min: float = 70.0
    f0_max: float = 400.0
    voicing_threshold: float = 0.45
    version: str = "logmel40-acfpitch-0.1"

    @property
    def win(self) -> int:
        return int(round(self.sample_rate * self.win_ms / 1000))

    @property
    def hop(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000))

    @property
    def dim(self) -> int:
        return self.n_mels + (2 if self.pitch else 0)

    @property
    def frames_per_second(self) -> float:
        return 1000.0 / self.hop_ms

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> FeatureConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> np.ndarray:
    """Triangular HTK-mel filters, shape [n_mels, n_fft // 2 + 1]."""
    n_bins = cfg.n_fft // 2 + 1
    bin_hz = np.linspace(0.0, cfg.sample_rate / 2, n_bins)
    mels = np.linspace(_hz_to_mel(cfg.fmin), _hz_to_mel(cfg.fmax), cfg.n_mels + 2)
    edges = _mel_to_hz(mels)
    fb = np.zeros((cfg.n_mels, n_bins), dtype=np.float64)
    for m in range(cfg.n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (bin_hz - lo) / max(mid - lo, 1e-9)
        down = (hi - bin_hz) / max(hi - mid, 1e-9)
        fb[m] = np.clip(np.minimum(up, down), 0.0, None)
    return fb.astype(np.float32)


class LogMelPitch:
    """Stateless per-window feature computation shared by batch and streaming paths."""

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()
        c = self.cfg
        if c.n_fft < c.win + int(np.ceil(c.sample_rate / c.f0_min)):
            raise ValueError("n_fft too small for linear (non-circular) autocorrelation")
        n = np.arange(c.win)
        self._window = (0.5 - 0.5 * np.cos(2 * np.pi * n / c.win)).astype(
            np.float32
        )  # periodic Hann
        self._fbank = mel_filterbank(c)
        self._lag_min = int(np.floor(c.sample_rate / c.f0_max))
        self._lag_max = int(np.ceil(c.sample_rate / c.f0_min))
        w = self._window.astype(np.float64)
        wacf = np.array([np.dot(w[: c.win - lag], w[lag:]) for lag in range(self._lag_max + 2)])
        self._wacf_norm = (wacf / wacf[0]).astype(np.float32)

    @property
    def dim(self) -> int:
        return self.cfg.dim

    def compute_windows(self, windows: np.ndarray) -> np.ndarray:
        """windows: [N, win] float → features [N, dim] float32."""
        c = self.cfg
        if windows.shape[0] == 0:
            return np.zeros((0, c.dim), dtype=np.float32)
        xw = windows.astype(np.float32) * self._window[None, :]
        spec = np.fft.rfft(xw, n=c.n_fft, axis=1)
        power = (spec.real**2 + spec.imag**2).astype(np.float32)
        logmel = np.log(power @ self._fbank.T + 1e-6)
        if not c.pitch:
            return logmel.astype(np.float32)
        acf = np.fft.irfft(power, n=c.n_fft, axis=1)[:, : self._lag_max + 2]
        r0 = acf[:, :1] + 1e-9
        nacf = acf / r0 / np.maximum(self._wacf_norm[None, :], 1e-3)
        seg = nacf[:, self._lag_min : self._lag_max + 1]
        # first-peak picking avoids sub-harmonic (octave-down) errors: take the smallest lag
        # that is a local maximum within 85% of the global peak
        gmax = seg.max(axis=1, keepdims=True)
        local = np.zeros_like(seg, dtype=bool)
        local[:, 1:-1] = (seg[:, 1:-1] >= seg[:, :-2]) & (seg[:, 1:-1] >= seg[:, 2:])
        cand = local & (seg >= 0.85 * gmax)
        idx = np.where(cand.any(axis=1), np.argmax(cand, axis=1), np.argmax(seg, axis=1))
        peak = seg[np.arange(seg.shape[0]), idx]
        # parabolic interpolation around the peak for a sub-sample lag
        left = seg[np.arange(seg.shape[0]), np.clip(idx - 1, 0, seg.shape[1] - 1)]
        right = seg[np.arange(seg.shape[0]), np.clip(idx + 1, 0, seg.shape[1] - 1)]
        denom = left - 2 * peak + right
        safe = np.where(np.abs(denom) > 1e-9, denom, -1.0)
        offset = np.where(np.abs(denom) > 1e-9, 0.5 * (left - right) / safe, 0.0)
        lag = self._lag_min + idx + np.clip(offset, -0.5, 0.5)
        periodicity = np.clip(peak, 0.0, 1.0)
        energy_ok = r0[:, 0] > 1e-7 * c.win
        voiced = (periodicity > c.voicing_threshold) & energy_ok
        logf0 = np.where(voiced, np.log2(c.sample_rate / lag / 100.0), 0.0)
        periodicity = np.where(energy_ok, periodicity, 0.0)
        return np.concatenate([logmel, periodicity[:, None], logf0[:, None]], axis=1).astype(
            np.float32
        )

    def batch(self, audio: np.ndarray, frame_ms: float = 20.0, chunk: int = 20_000) -> np.ndarray:
        """Features for a whole signal: ``(frame_ms / hop_ms)`` rows per complete audio frame."""
        c = self.cfg
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        frame_samples = int(round(c.sample_rate * frame_ms / 1000))
        n_frames = audio.size // frame_samples
        per_frame = frame_samples // c.hop
        n_windows = n_frames * per_frame
        if n_windows == 0:
            return np.zeros((0, c.dim), dtype=np.float32)
        padded = np.concatenate([np.zeros(c.win - c.hop, dtype=np.float32), audio])
        view = np.lib.stride_tricks.sliding_window_view(padded, c.win)[:: c.hop][:n_windows]
        out = np.empty((n_windows, c.dim), dtype=np.float32)
        for i in range(0, n_windows, chunk):
            out[i : i + chunk] = self.compute_windows(view[i : i + chunk])
        return out


class StreamingFeatures:
    """Incremental features: push 20 ms frames, receive the newly completed windows."""

    def __init__(self, extractor: LogMelPitch) -> None:
        self.extractor = extractor
        self.reset()

    def reset(self) -> None:
        c = self.extractor.cfg
        self._buf = np.zeros(c.win - c.hop, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        c = self.extractor.cfg
        buf = np.concatenate([self._buf, np.asarray(samples, dtype=np.float32).reshape(-1)])
        if buf.size < c.win:
            self._buf = buf
            return np.zeros((0, c.dim), dtype=np.float32)
        n = (buf.size - c.win) // c.hop + 1
        windows = np.lib.stride_tricks.sliding_window_view(buf, c.win)[:: c.hop][:n]
        feats = self.extractor.compute_windows(windows)
        self._buf = buf[n * c.hop :].copy()
        return feats
