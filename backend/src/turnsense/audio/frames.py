"""Timestamped audio frames and a bounded, gap-aware re-framing buffer (architecture.md §4.5)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_FRAME_MS = 20


@dataclass(frozen=True, slots=True)
class AudioFrame:
    """A fixed-size mono float32 frame in [-1, 1].

    ``start_ms``/``end_ms`` are session-monotonic times derived from the sample count,
    so they are exact and never drift; ``received_ms`` is when the last sample arrived.
    """

    seq: int
    start_ms: float
    end_ms: float
    samples: np.ndarray
    sample_rate: int
    received_ms: float | None = None

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass(frozen=True, slots=True)
class FrameGap:
    """A detected discontinuity. Recorded rather than papered over with fabricated audio."""

    after_seq: int
    missing_ms: float
    at_ms: float


def pcm16_to_float(pcm: bytes | np.ndarray) -> np.ndarray:
    arr = np.frombuffer(pcm, dtype=np.int16) if isinstance(pcm, bytes | bytearray) else pcm
    return arr.astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resampler; adequate for VAD/turn features, not for playback quality."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = int(round(samples.size * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-6))


class FrameBuffer:
    """Re-frames arbitrary chunks into fixed frames with exact sample-derived timestamps.

    * Incoming chunks may have any length/sample rate; output frames are ``frame_ms`` long
      at ``sample_rate`` mono.
    * Callers may pass the chunk's capture time; if it is later than the buffer's sample
      clock by more than ``gap_tolerance_ms`` the gap is recorded and the clock jumps
      forward (no zero-filled audio is invented).
    * Complete frames are returned immediately, so at most one partial frame is held here;
      bounding the queue of frames awaiting processing is the consumer's job
      (see ``turnsense.pipeline.FrameQueue``).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        gap_tolerance_ms: float = 40.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self._gap_tolerance_ms = gap_tolerance_ms
        self._pending = np.zeros(0, dtype=np.float32)
        self._clock_samples = 0  # samples emitted or skipped so far
        self._seq = 0
        self.gaps: deque[FrameGap] = deque(maxlen=1000)
        self.dropped_samples = 0
        self.total_gap_ms = 0.0

    @property
    def clock_ms(self) -> float:
        """Session time at the end of the audio that has been buffered so far."""
        return (self._clock_samples + self._pending.size) * 1000.0 / self.sample_rate

    def push(
        self,
        samples: np.ndarray,
        sample_rate: int | None = None,
        capture_ms: float | None = None,
        received_ms: float | None = None,
    ) -> list[AudioFrame]:
        """Add a chunk; ``capture_ms`` is the session time of the chunk's first sample."""
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        src_rate = sample_rate or self.sample_rate
        chunk = resample_linear(samples.astype(np.float32, copy=False), src_rate, self.sample_rate)

        if capture_ms is not None:
            expected = self.clock_ms
            if capture_ms - expected > self._gap_tolerance_ms:
                self._record_gap(capture_ms - expected, expected)

        self._pending = np.concatenate([self._pending, chunk]) if self._pending.size else chunk

        frames: list[AudioFrame] = []
        while self._pending.size >= self.frame_samples:
            data = self._pending[: self.frame_samples].copy()
            self._pending = self._pending[self.frame_samples :]
            start = self._clock_samples * 1000.0 / self.sample_rate
            self._clock_samples += self.frame_samples
            self._seq += 1
            frames.append(
                AudioFrame(
                    seq=self._seq,
                    start_ms=start,
                    end_ms=start + self.frame_ms,
                    samples=data,
                    sample_rate=self.sample_rate,
                    received_ms=received_ms,
                )
            )
        return frames

    def _record_gap(self, missing_ms: float, at_ms: float) -> None:
        # Discard the partial frame straddling the gap and jump the clock forward.
        skip = int(round(missing_ms * self.sample_rate / 1000))
        self._clock_samples += self._pending.size + skip
        self.dropped_samples += self._pending.size
        self._pending = np.zeros(0, dtype=np.float32)
        self.total_gap_ms += missing_ms
        self.gaps.append(FrameGap(after_seq=self._seq, missing_ms=missing_ms, at_ms=at_ms))

    @property
    def dropped_frames(self) -> int:
        """Frames' worth of audio lost to transport gaps."""
        lost = self.dropped_samples + self.total_gap_ms * self.sample_rate / 1000
        return int(lost // self.frame_samples)
