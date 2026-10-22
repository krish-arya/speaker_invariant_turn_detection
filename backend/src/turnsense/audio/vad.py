"""Voice activity detection: speech presence only, never turn completion (architecture.md §4.6).

Two implementations share one interface:

* ``EnergyVAD`` — adaptive-noise-floor energy detector. Deterministic, dependency-free; used
  by the simulation transport and tests. It reports ``probability=None`` because it has no
  calibrated probability.
* ``SileroVAD`` — Silero VAD v5 through ONNX Runtime (the project's runtime VAD). It consumes
  512-sample windows at 16 kHz, so a 20 ms frame carries the most recent window's probability.

Both apply the same hysteresis and track run lengths (current speech run / silence since the
last speech) that detectors consume.
"""

from __future__ import annotations

import importlib.resources
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from turnsense.audio.frames import AudioFrame, rms_dbfs
from turnsense.events import VadConfig


@dataclass(frozen=True, slots=True)
class VadResult:
    frame_seq: int
    frame_end_ms: float
    is_speech: bool  # smoothed (hysteresis) speech state
    raw_speech: bool  # this frame's unsmoothed decision
    probability: float | None  # None when the VAD has no calibrated probability
    speech_run_ms: float  # length of the current speech run (0 when silent)
    silence_run_ms: float  # silence since the last speech ended (0 while speaking)
    last_speech_end_ms: float | None  # end time of the most recent speech frame
    changed: bool  # smoothed state flipped on this frame


class VoiceActivityDetector(Protocol):
    kind: str

    def process(self, frame: AudioFrame) -> VadResult: ...

    def reset(self) -> None: ...


class _Hysteresis:
    """Turns raw per-frame decisions into a stable speech state plus run lengths."""

    def __init__(self, min_speech_frames: int, min_silence_frames: int) -> None:
        self._min_on = max(1, min_speech_frames)
        self._min_off = max(1, min_silence_frames)
        self.reset()

    def reset(self) -> None:
        self._speaking = False
        self._on_count = 0
        self._off_count = 0
        self._speech_start_ms: float | None = None
        self._last_speech_end_ms: float | None = None

    def update(self, frame: AudioFrame, raw: bool, prob: float | None) -> VadResult:
        changed = False
        if raw:
            self._on_count += 1
            self._off_count = 0
            if not self._speaking and self._on_count >= self._min_on:
                self._speaking = True
                changed = True
                # speech began at the first frame of the qualifying run
                self._speech_start_ms = frame.end_ms - self._on_count * frame.duration_ms
            if self._speaking:
                self._last_speech_end_ms = frame.end_ms
        else:
            self._off_count += 1
            self._on_count = 0
            if self._speaking and self._off_count >= self._min_off:
                self._speaking = False
                changed = True
                self._speech_start_ms = None

        if self._speaking and self._speech_start_ms is not None:
            speech_run = frame.end_ms - self._speech_start_ms
            silence_run = 0.0
        else:
            speech_run = 0.0
            silence_run = (
                frame.end_ms - self._last_speech_end_ms
                if self._last_speech_end_ms is not None
                else 0.0
            )
        return VadResult(
            frame_seq=frame.seq,
            frame_end_ms=frame.end_ms,
            is_speech=self._speaking,
            raw_speech=raw,
            probability=prob,
            speech_run_ms=speech_run,
            silence_run_ms=silence_run,
            last_speech_end_ms=self._last_speech_end_ms,
            changed=changed,
        )


class EnergyVAD:
    """Energy detector with a slowly adapting noise floor."""

    kind = "energy"

    def __init__(
        self,
        config: VadConfig | None = None,
        margin_db: float = 12.0,
        absolute_floor_dbfs: float = -55.0,
        floor_rise_db_per_s: float = 3.0,
    ) -> None:
        cfg = config or VadConfig(kind="energy")
        self._margin = margin_db
        self._abs_floor = absolute_floor_dbfs
        self._rise = floor_rise_db_per_s
        self._hyst = _Hysteresis(cfg.min_speech_frames, cfg.min_silence_frames)
        self.reset()

    def reset(self) -> None:
        self._noise_floor: float | None = None
        self._hyst.reset()

    def process(self, frame: AudioFrame) -> VadResult:
        level = rms_dbfs(frame.samples)
        if self._noise_floor is None:
            self._noise_floor = min(level, self._abs_floor - 10.0)
        elif level < self._noise_floor:
            self._noise_floor = 0.9 * self._noise_floor + 0.1 * level  # fall fast
        else:  # rise slowly so sustained speech is not absorbed into the floor
            self._noise_floor += self._rise * frame.duration_ms / 1000.0
        raw = level > max(self._noise_floor + self._margin, self._abs_floor)
        return self._hyst.update(frame, raw, None)


def find_silero_model() -> Path | None:
    """Locate silero_vad.onnx: $TURNSENSE_SILERO_ONNX, then the LiveKit Silero plugin bundle."""
    env = os.environ.get("TURNSENSE_SILERO_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    try:
        res = importlib.resources.files("livekit.plugins.silero") / "resources" / "silero_vad.onnx"
        if res.is_file():
            return Path(str(res))
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    return None


class SileroVAD:
    """Silero VAD v5 via ONNX Runtime, streaming over 20 ms frames."""

    kind = "silero"
    WINDOW = 512  # samples at 16 kHz
    CONTEXT = 64

    def __init__(self, config: VadConfig | None = None, model_path: Path | None = None) -> None:
        import onnxruntime as ort  # optional dependency: pip install turnsense[vad]

        path = model_path or find_silero_model()
        if path is None:
            raise FileNotFoundError(
                "silero_vad.onnx not found; set TURNSENSE_SILERO_ONNX or install livekit-plugins-silero"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=opts
        )
        cfg = config or VadConfig(kind="silero")
        self._threshold = cfg.threshold
        self._hyst = _Hysteresis(cfg.min_speech_frames, cfg.min_silence_frames)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT), dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._prob = 0.0
        self._hyst.reset()

    def process(self, frame: AudioFrame) -> VadResult:
        if frame.sample_rate != 16000:
            raise ValueError("SileroVAD expects 16 kHz frames")
        self._pending = np.concatenate([self._pending, frame.samples])
        while self._pending.size >= self.WINDOW:
            window = self._pending[: self.WINDOW].reshape(1, -1)
            self._pending = self._pending[self.WINDOW :]
            x = np.concatenate([self._context, window], axis=1)
            out, self._state = self._session.run(
                None,
                {"input": x, "state": self._state, "sr": np.array(16000, dtype=np.int64)},
            )
            self._context = x[:, -self.CONTEXT :]
            self._prob = float(np.asarray(out).reshape(-1)[0])
        return self._hyst.update(frame, self._prob >= self._threshold, self._prob)


def create_vad(config: VadConfig) -> VoiceActivityDetector:
    if config.kind == "silero":
        return SileroVAD(config)
    return EnergyVAD(config)


def silero_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return find_silero_model() is not None
