"""A virtual-time asyncio event loop for faster-than-real-time scripted trials.

``loop.time()`` returns a virtual clock that jumps straight to the next scheduled timer
whenever there is no ready I/O, so ``asyncio.sleep`` costs no wall time. All pipeline timing
(frame pacing, modelled provider latency, playback duration) is expressed through the loop
clock, so a trial run here produces the same event timeline as a real-time run of the same
seed — minus real-time scheduling jitter.

Real CPU time spent inside callbacks (e.g. detector inference) does not advance virtual time;
it is measured separately with ``perf_counter`` and reported as detector inference latency.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")


class _VirtualSelector(selectors.BaseSelector):
    def __init__(self) -> None:
        self._real = selectors.DefaultSelector()
        self.now = 0.0

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        events = self._real.select(0)
        if events:
            return events
        if timeout is None:  # nothing scheduled: block briefly on real I/O (thread wake-ups)
            return self._real.select(0.01)
        self.now += max(timeout, 0.0)
        return []

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._real.unregister(fileobj)

    def modify(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._real.modify(fileobj, events, data)

    def get_key(self, fileobj: Any) -> selectors.SelectorKey:
        return self._real.get_key(fileobj)

    def get_map(self) -> Any:
        return self._real.get_map()

    def close(self) -> None:
        self._real.close()


class VirtualTimeEventLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self._vsel = _VirtualSelector()
        super().__init__(self._vsel)

    def time(self) -> float:
        return self._vsel.now


def run_in_virtual_time(factory: Callable[[], Awaitable[T]]) -> T:
    """Run a coroutine to completion on a fresh virtual-time loop (call from a worker thread)."""
    loop = VirtualTimeEventLoop()
    try:
        return loop.run_until_complete(factory())
    finally:
        loop.close()
