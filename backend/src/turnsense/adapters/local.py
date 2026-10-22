"""Local/open fallbacks for the reproducible benchmark profile (architecture.md §11, Profile B).

* ``WhisperSTT`` — faster-whisper is not a streaming recogniser: audio is buffered and a final
  hypothesis is produced when the orchestrator commits a turn end (``finalize``). There are
  no partials, so the transcript appears only after the turn decision.
* ``KokoroTTS`` — Kokoro-82M synthesis per clause in a worker thread.

Both need the ``local`` extra (``pip install -e ".[local]"``) and are imported lazily.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np

from turnsense.adapters.base import (
    ErrorCallback,
    ProviderError,
    STTAdapter,
    SttCallback,
    SttEvent,
    TTSAdapter,
)
from turnsense.audio.frames import AudioFrame
from turnsense.events import ErrorSource

log = logging.getLogger(__name__)


class WhisperSTT(STTAdapter):
    def __init__(self, model: str = "small.en", max_buffer_s: float = 30.0) -> None:
        from faster_whisper import WhisperModel

        self.name = f"faster_whisper_{model.replace('.', '_')}"
        self._model = WhisperModel(model, device="cpu", compute_type="int8")
        self._max = int(16_000 * max_buffer_s)
        self._buf: list[np.ndarray] = []
        self._buf_start_ms: float | None = None
        self._buf_end_ms = 0.0
        self._pending: asyncio.Task[None] | None = None

    async def start(self, on_event: SttCallback, on_error: ErrorCallback) -> None:
        self._on_event, self._on_error = on_event, on_error

    def push_frame(self, frame: AudioFrame) -> None:
        if self._buf_start_ms is None:
            self._buf_start_ms = frame.start_ms
        self._buf.append(frame.samples)
        self._buf_end_ms = frame.end_ms
        total = sum(b.size for b in self._buf)
        while total > self._max and self._buf:
            total -= self._buf.pop(0).size
            self._buf_start_ms = self._buf_end_ms - total / 16.0

    def finalize(self) -> None:
        if not self._buf:
            return
        audio = np.concatenate(self._buf)
        start, end = self._buf_start_ms, self._buf_end_ms
        self._buf, self._buf_start_ms = [], None
        self._pending = asyncio.create_task(self._transcribe(audio, start, end))

    async def _transcribe(self, audio: np.ndarray, start: float | None, end: float) -> None:
        def run() -> str:
            segments, _ = self._model.transcribe(audio, beam_size=1, vad_filter=True, language="en")
            return " ".join(s.text.strip() for s in segments).strip()

        try:
            text = await asyncio.to_thread(run)
        except Exception as exc:
            self._on_error(ProviderError(ErrorSource.STT, "whisper_failed", str(exc)))
            return
        if text:
            self._on_event(SttEvent(text, True, start, end))

    async def close(self) -> None:
        if self._pending:
            self._pending.cancel()


class KokoroTTS(TTSAdapter):
    def __init__(self, voice: str = "af_heart", lang_code: str = "a") -> None:
        from kokoro import KPipeline

        self.name = "kokoro_82m"
        self.sample_rate = 24_000
        self._pipe = KPipeline(lang_code=lang_code)
        self._voice = voice

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        def run() -> list[np.ndarray]:
            out = []
            for _, _, audio in self._pipe(text, voice=self._voice):
                out.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            return out

        try:
            chunks = await asyncio.to_thread(run)
        except Exception as exc:
            raise ProviderError(ErrorSource.TTS, "kokoro_failed", str(exc)) from exc
        step = self.sample_rate // 5  # 200 ms chunks so cancellation stays responsive
        for chunk in chunks:
            for i in range(0, chunk.size, step):
                yield chunk[i : i + step]
