"""Deepgram adapters: Nova-3 streaming STT and Aura-2 streaming TTS over WebSocket.

The API key stays in the worker process. Message handling is factored into pure functions
(``parse_stt_message``) so it can be unit-tested without network access.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import numpy as np

from turnsense.adapters.base import (
    ErrorCallback,
    ProviderError,
    STTAdapter,
    SttCallback,
    SttEvent,
    TTSAdapter,
)
from turnsense.audio.frames import AudioFrame, float_to_pcm16, pcm16_to_float
from turnsense.events import ErrorSource

log = logging.getLogger(__name__)

STT_URL = "wss://api.deepgram.com/v1/listen"
TTS_URL = "wss://api.deepgram.com/v1/speak"


def parse_stt_message(msg: dict[str, Any], offset_ms: float = 0.0) -> SttEvent | None:
    """Map a Deepgram ``Results`` message to an SttEvent; other message types → None."""
    if msg.get("type") != "Results":
        return None
    alts = (msg.get("channel") or {}).get("alternatives") or []
    text = (alts[0].get("transcript") if alts else "") or ""
    if not text.strip():
        return None
    start = float(msg.get("start", 0.0)) * 1000 + offset_ms
    end = start + float(msg.get("duration", 0.0)) * 1000
    return SttEvent(
        text=text.strip(), is_final=bool(msg.get("is_final")), start_ms=start, end_ms=end
    )


class DeepgramSTT(STTAdapter):
    def __init__(
        self, api_key: str, model: str = "nova-3", sample_rate: int = 16_000, language: str = "en"
    ) -> None:
        self.name = f"deepgram_{model.replace('-', '_')}"
        self._key = api_key
        self._params = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": sample_rate,
            "channels": 1,
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
            "language": language,
            # turn-taking is our detector's job; Deepgram only transcribes
            "endpointing": "false",
            "vad_events": "false",
        }
        self._out: asyncio.Queue[bytes | str | None] = asyncio.Queue(maxsize=500)
        self._tasks: list[asyncio.Task[None]] = []
        self._offset_ms: float | None = None
        self._closed = False

    async def start(self, on_event: SttCallback, on_error: ErrorCallback) -> None:
        from websockets.asyncio.client import connect

        url = f"{STT_URL}?{urlencode(self._params)}"
        try:
            self._ws = await connect(
                url, additional_headers={"Authorization": f"Token {self._key}"}, max_size=None
            )
        except Exception as exc:
            raise ProviderError(
                ErrorSource.STT, "connect_failed", f"Deepgram STT connection failed: {exc}"
            ) from exc
        self._tasks = [
            asyncio.create_task(self._send_loop(on_error)),
            asyncio.create_task(self._recv_loop(on_event, on_error)),
        ]

    def push_frame(self, frame: AudioFrame) -> None:
        if self._closed:
            return
        if self._offset_ms is None:
            self._offset_ms = frame.start_ms  # Deepgram times are relative to the first byte sent
        try:
            self._out.put_nowait(float_to_pcm16(frame.samples))
        except asyncio.QueueFull:  # never block the frame loop; the gap shows up in STT only
            log.warning("deepgram stt send queue full; dropping audio")

    def finalize(self) -> None:
        with contextlib.suppress(asyncio.QueueFull):
            self._out.put_nowait(json.dumps({"type": "Finalize"}))

    async def _send_loop(self, on_error: ErrorCallback) -> None:
        try:
            while True:
                try:
                    item = await asyncio.wait_for(self._out.get(), timeout=5.0)
                except TimeoutError:
                    item = json.dumps({"type": "KeepAlive"})
                if item is None:
                    await self._ws.send(json.dumps({"type": "CloseStream"}))
                    return
                await self._ws.send(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                on_error(
                    ProviderError(
                        ErrorSource.STT, "send_failed", f"Deepgram STT send failed: {exc}"
                    )
                )

    async def _recv_loop(self, on_event: SttCallback, on_error: ErrorCallback) -> None:
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                if msg.get("type") == "Error" or "err_code" in msg:
                    on_error(
                        ProviderError(
                            ErrorSource.STT,
                            str(msg.get("err_code", "provider_error")),
                            str(msg.get("err_msg", msg)),
                        )
                    )
                    continue
                ev = parse_stt_message(msg, self._offset_ms or 0.0)
                if ev is not None:
                    on_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                on_error(
                    ProviderError(
                        ErrorSource.STT, "connection_lost", f"Deepgram STT connection lost: {exc}"
                    )
                )

    async def close(self) -> None:
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self._out.put_nowait(None)
        await asyncio.sleep(0)
        for t in self._tasks:
            t.cancel()
        with contextlib.suppress(Exception):
            await self._ws.close()


class DeepgramTTS(TTSAdapter):
    """Aura-2 over one WebSocket per session; one Speak+Flush per clause.

    If a synthesis is cancelled mid-clause (barge-in), the socket is closed so its remaining
    audio cannot be mistaken for the next clause; the next call reconnects.
    """

    def __init__(
        self, api_key: str, voice: str = "aura-2-thalia-en", sample_rate: int = 24_000
    ) -> None:
        self.name = f"deepgram_{voice.replace('-', '_')}"
        self.sample_rate = sample_rate
        self._key = api_key
        self._url = f"{TTS_URL}?{urlencode({'model': voice, 'encoding': 'linear16', 'sample_rate': sample_rate})}"
        self._ws: Any = None
        self._lock = asyncio.Lock()

    async def _connect(self) -> Any:
        from websockets.asyncio.client import connect

        if self._ws is None:
            try:
                self._ws = await connect(
                    self._url,
                    additional_headers={"Authorization": f"Token {self._key}"},
                    max_size=None,
                )
            except Exception as exc:
                raise ProviderError(
                    ErrorSource.TTS, "connect_failed", f"Deepgram TTS connection failed: {exc}"
                ) from exc
        return self._ws

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        async with self._lock:
            ws = await self._connect()
            complete = False
            try:
                await ws.send(json.dumps({"type": "Speak", "text": text}))
                await ws.send(json.dumps({"type": "Flush"}))
                async for raw in ws:
                    if isinstance(raw, bytes):
                        if raw:
                            yield pcm16_to_float(raw)
                        continue
                    msg = json.loads(raw)
                    kind = msg.get("type")
                    if kind == "Flushed":
                        complete = True
                        return
                    if kind == "Error":
                        raise ProviderError(
                            ErrorSource.TTS,
                            str(msg.get("err_code", "provider_error")),
                            str(msg.get("err_msg", msg)),
                        )
                raise ProviderError(
                    ErrorSource.TTS, "connection_closed", "Deepgram TTS closed mid-utterance"
                )
            except ProviderError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ProviderError(
                    ErrorSource.TTS, "stream_failed", f"Deepgram TTS failed: {exc}"
                ) from exc
            finally:
                if not complete:
                    await self._drop()

    async def close(self) -> None:
        await self._drop()
