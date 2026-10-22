"""Provider-agnostic STT / LLM / TTS / playback interfaces (architecture.md §2.3, §4.8–4.10)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from turnsense.audio.frames import AudioFrame
from turnsense.events import ErrorSource


@dataclass(frozen=True, slots=True)
class SttEvent:
    text: str
    is_final: bool
    start_ms: float | None = None  # session-time span of the audio, when the provider says
    end_ms: float | None = None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class ProviderError(Exception):
    """A provider/network failure. Surfaced as a recoverable error, never as a turn event."""

    def __init__(
        self, source: ErrorSource, code: str, message: str, recoverable: bool = True
    ) -> None:
        super().__init__(message)
        self.source = source
        self.code = code
        self.recoverable = recoverable


SttCallback = Callable[[SttEvent], None]
ErrorCallback = Callable[[ProviderError], None]


class STTAdapter(ABC):
    name: str

    @abstractmethod
    async def start(self, on_event: SttCallback, on_error: ErrorCallback) -> None: ...

    @abstractmethod
    def push_frame(self, frame: AudioFrame) -> None:
        """Non-blocking; adapters buffer/forward internally."""

    def finalize(self) -> None:  # noqa: B027 - optional hook
        """The orchestrator committed a turn end: flush a final hypothesis as soon as possible."""

    @abstractmethod
    async def close(self) -> None: ...


class LLMAdapter(ABC):
    name: str

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Yield text deltas. Cancelling the consuming task must stop the request."""


class TTSAdapter(ABC):
    name: str
    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield mono float32 audio chunks at ``sample_rate``."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        pass


class PlaybackSink(ABC):
    """Where agent audio goes: a LiveKit AudioSource, or a timed null sink in simulation."""

    @abstractmethod
    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Deliver one chunk; returns when the sink has accepted it."""

    @abstractmethod
    def flush(self) -> None:
        """Discard anything queued but not yet audible."""

    async def drain(self) -> None:  # noqa: B027 - optional hook
        """Wait until everything delivered has been played out."""
