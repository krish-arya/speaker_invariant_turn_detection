"""Deterministic simulated providers for scripted trials and tests.

Latencies are drawn from seeded distributions and realised with ``asyncio.sleep`` so they run
identically in real time and in virtual time. They model a hosted pipeline's *shape*; they are
not measurements of any provider. Trial records label these runs ``runtime_profile=simulation``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

import numpy as np

from turnsense.adapters.base import (
    ChatMessage,
    ErrorCallback,
    LLMAdapter,
    PlaybackSink,
    STTAdapter,
    SttCallback,
    SttEvent,
    TTSAdapter,
)
from turnsense.audio.frames import AudioFrame
from turnsense.sim.voice import VoiceProfile, synth_phrase

AGENT_VOICE_ID = 9_999


@dataclass(frozen=True, slots=True)
class ScriptWord:
    text: str
    start_ms: float
    end_ms: float
    phrase_id: int


class MockSTT(STTAdapter):
    """Emits partials as words finish and a final per phrase, driven by the audio clock."""

    name = "mock_stt"

    def __init__(
        self,
        words: Sequence[ScriptWord],
        partial_lag_ms: float = 120.0,
        final_lag_ms: float = 280.0,
    ) -> None:
        self._words = sorted(words, key=lambda w: w.start_ms)
        self._partial_lag = partial_lag_ms
        self._final_lag = final_lag_ms
        self._phrases: dict[int, list[ScriptWord]] = {}
        for w in self._words:
            self._phrases.setdefault(w.phrase_id, []).append(w)
        self._partial_sent: dict[int, int] = {}
        self._final_sent: set[int] = set()
        self._on_event: SttCallback | None = None

    async def start(self, on_event: SttCallback, on_error: ErrorCallback) -> None:
        self._on_event = on_event

    def push_frame(self, frame: AudioFrame) -> None:
        if self._on_event is None:
            return
        now = frame.end_ms
        for pid, words in self._phrases.items():
            if pid in self._final_sent:
                continue
            if words[-1].end_ms + self._final_lag <= now:
                self._final_sent.add(pid)
                self._on_event(
                    SttEvent(
                        " ".join(w.text for w in words), True, words[0].start_ms, words[-1].end_ms
                    )
                )
                continue
            heard = sum(1 for w in words if w.end_ms + self._partial_lag <= now)
            if heard and heard != self._partial_sent.get(pid):
                self._partial_sent[pid] = heard
                span = words[:heard]
                self._on_event(
                    SttEvent(
                        " ".join(w.text for w in span), False, span[0].start_ms, span[-1].end_ms
                    )
                )

    async def close(self) -> None:
        self._on_event = None


_SENTENCES = [
    "Sure, I can help with that.",
    "The short answer is yes, as long as the booking is confirmed before noon.",
    "There are two options, a direct route and a cheaper one with a single stop.",
    "Most people choose the morning departure because it arrives in time for lunch.",
    "If you change your mind later, the fare can usually be adjusted for a small fee.",
    "I would also suggest checking the weather, since it might rain in the evening.",
    "Let me know which option sounds better and I will take care of the rest.",
    "That is a good question, and it depends a little on your schedule.",
]


class MockLLM(LLMAdapter):
    name = "mock_llm"

    def __init__(
        self,
        seed: int,
        first_token_ms: tuple[float, float] = (280.0, 50.0),
        token_ms: tuple[float, float] = (24.0, 6.0),
        sentences: tuple[int, int] = (2, 3),
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._first = first_token_ms
        self._tok = token_ms
        self._sentences = sentences

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        await asyncio.sleep(max(0.03, self._rng.normal(*self._first)) / 1000)
        n = int(self._rng.integers(self._sentences[0], self._sentences[1] + 1))
        picks = self._rng.choice(len(_SENTENCES), size=n, replace=False)
        words = " ".join(_SENTENCES[i] for i in picks).split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word
            await asyncio.sleep(max(0.002, self._rng.normal(*self._tok)) / 1000)


class MockTTS(TTSAdapter):
    name = "mock_tts"

    def __init__(
        self,
        seed: int,
        sample_rate: int = 16_000,
        first_audio_ms: tuple[float, float] = (180.0, 35.0),
        real_time_factor: float = 0.3,
        chunk_ms: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self._rng = np.random.default_rng(seed + 1)
        self._voice = VoiceProfile.from_id(AGENT_VOICE_ID)
        self._first = first_audio_ms
        self._rtf = real_time_factor
        self._chunk = sample_rate * chunk_ms // 1000

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        await asyncio.sleep(max(0.02, self._rng.normal(*self._first)) / 1000)
        n_syll = max(2, round(len(text.split()) * 1.4))
        contour = "final" if text.rstrip().endswith((".", "!", "?")) else "hold"
        audio = synth_phrase(self._voice, n_syll, contour, self._rng, self.sample_rate)
        for i in range(0, audio.size, self._chunk):
            chunk = audio[i : i + self._chunk]
            yield chunk
            await asyncio.sleep(chunk.size / self.sample_rate * self._rtf)


class TimedNullSink(PlaybackSink):
    """Simulated playout: accepting a chunk takes as long as the chunk lasts."""

    def __init__(self) -> None:
        self.delivered_samples = 0

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self.delivered_samples += samples.size
        await asyncio.sleep(samples.size / sample_rate)

    def flush(self) -> None:
        pass
