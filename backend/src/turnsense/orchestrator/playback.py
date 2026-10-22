"""Generation-aware playback queue (architecture.md §4.10, §9).

Every chunk carries the response generation ID that produced it. The controller only plays
chunks of the *current, non-cancelled* generation; anything else is rejected and counted, so
audio from a cancelled response can never leak into the next one.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from turnsense.adapters.base import PlaybackSink


@dataclass(slots=True)
class _Item:
    generation_id: int
    samples: np.ndarray | None  # None marks end-of-generation
    sample_rate: int
    clause: str | None


class PlaybackController:
    def __init__(
        self,
        sink: PlaybackSink,
        *,
        on_first_audio: Callable[[int], None],
        on_clause: Callable[[int, str], None],
        on_drained: Callable[[int], None],
        on_stale: Callable[[int], None],
        on_chunk: Callable[[int, np.ndarray], None] | None = None,
        on_error: Callable[[int, BaseException], None] | None = None,
    ) -> None:
        self._sink = sink
        self._on_first_audio = on_first_audio
        self._on_clause = on_clause
        self._on_drained = on_drained
        self._on_stale = on_stale
        self._on_chunk = on_chunk
        self._on_error = on_error
        self._queue: asyncio.Queue[_Item] = asyncio.Queue()
        self._current: int | None = None
        self._cancelled: set[int] = set()
        self._started: set[int] = set()
        self._play_task: asyncio.Task[None] | None = None
        self._playing_gen: int | None = None
        self._runner: asyncio.Task[None] | None = None
        self.stale_rejected = 0
        self.stale_played = 0  # must stay 0; counted defensively at the delivery point
        self.chunks_played = 0

    def start(self) -> None:
        self._runner = asyncio.create_task(self._run(), name="playback")

    async def stop(self) -> None:
        if self._current is not None:
            self.cancel(self._current)
        if self._runner:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner

    def begin(self, generation_id: int) -> None:
        self._current = generation_id

    def is_live(self, generation_id: int) -> bool:
        return generation_id == self._current and generation_id not in self._cancelled

    def put(
        self, generation_id: int, samples: np.ndarray, sample_rate: int, clause: str | None = None
    ) -> bool:
        if not self.is_live(generation_id):
            self.stale_rejected += 1
            self._on_stale(generation_id)
            return False
        self._queue.put_nowait(_Item(generation_id, samples, sample_rate, clause))
        return True

    def end_of_generation(self, generation_id: int) -> None:
        if self.is_live(generation_id):
            self._queue.put_nowait(_Item(generation_id, None, 0, None))

    def cancel(self, generation_id: int) -> int:
        """Stop and discard everything for ``generation_id``. Returns queued chunks dropped."""
        self._cancelled.add(generation_id)
        if self._current == generation_id:
            self._current = None
        kept: list[_Item] = []
        dropped = 0
        while not self._queue.empty():
            item = self._queue.get_nowait()
            if item.generation_id == generation_id:
                dropped += item.samples is not None
            else:
                kept.append(item)
        for item in kept:
            self._queue.put_nowait(item)
        if self._playing_gen == generation_id and self._play_task and not self._play_task.done():
            self._play_task.cancel()
        self._sink.flush()
        return dropped

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            gen = item.generation_id
            if not self.is_live(gen):
                if item.samples is not None:
                    self.stale_rejected += 1
                    self._on_stale(gen)
                continue
            if item.samples is None:
                await self._sink.drain()
                if self.is_live(gen):
                    self._on_drained(gen)
                continue
            if gen not in self._started:
                self._started.add(gen)
                self._on_first_audio(gen)
            if item.clause:
                self._on_clause(gen, item.clause)
            if gen in self._cancelled:  # defensive: never expected
                self.stale_played += 1
            self._playing_gen = gen
            self.chunks_played += 1
            if self._on_chunk:
                self._on_chunk(gen, item.samples)
            self._play_task = asyncio.create_task(self._sink.play(item.samples, item.sample_rate))
            try:
                await asyncio.wait({self._play_task})
            except asyncio.CancelledError:
                self._play_task.cancel()
                raise
            finally:
                self._playing_gen = None
            task = self._play_task
            if not task.cancelled() and task.exception() is not None:
                self.cancel(gen)
                if self._on_error:
                    self._on_error(gen, task.exception())
