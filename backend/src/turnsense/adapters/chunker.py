"""Clause chunker: groups an LLM token stream into speakable units for streaming TTS."""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"[.!?;:]\s|[,—–]\s|\n")


class ClauseChunker:
    """Emit a clause at punctuation once it is long enough; force-split very long runs.

    The first clause uses a lower minimum so the first audio arrives sooner.
    """

    def __init__(
        self, first_min_chars: int = 12, min_chars: int = 40, max_chars: int = 180
    ) -> None:
        self._first_min = first_min_chars
        self._min = min_chars
        self._max = max_chars
        self._buf = ""
        self._emitted = 0

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            clause = self._take()
            if clause is None:
                break
            if clause:  # empty when a cut consumed only whitespace/punctuation
                out.append(clause)
        return out

    def flush(self) -> list[str]:
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def _take(self) -> str | None:
        minimum = self._first_min if self._emitted == 0 else self._min
        for m in _BOUNDARY.finditer(self._buf):
            if m.end() >= minimum:
                return self._cut(m.end())
        if len(self._buf) > self._max:
            space = self._buf.rfind(" ", 0, self._max)
            return self._cut(space if space > minimum else self._max)
        return None

    def _cut(self, idx: int) -> str:
        clause, self._buf = self._buf[:idx].strip(), self._buf[idx:]
        if clause:
            self._emitted += 1
        return clause
