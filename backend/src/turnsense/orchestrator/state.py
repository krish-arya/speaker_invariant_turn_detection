"""Session state machine (architecture.md §6). Transitions are explicit and validated."""

from __future__ import annotations

import logging
from collections.abc import Callable

from turnsense.events import SessionState as S

log = logging.getLogger(__name__)

_ALWAYS = {S.ENDING, S.ERROR, S.RECONNECTING}

ALLOWED: dict[S, set[S]] = {
    S.CONNECTING: {S.LISTENING},
    S.LISTENING: {S.USER_SPEAKING},
    S.USER_SPEAKING: {S.POSSIBLE_END, S.LISTENING},  # → listening: too short to be speech
    S.POSSIBLE_END: {S.USER_SPEAKING, S.GENERATING, S.LISTENING},
    S.GENERATING: {S.AGENT_SPEAKING, S.INTERRUPTED, S.LISTENING},  # → listening: no reply
    S.AGENT_SPEAKING: {S.LISTENING, S.INTERRUPTED},
    S.INTERRUPTED: {S.USER_SPEAKING, S.LISTENING},
    S.RECONNECTING: {S.LISTENING},
    S.ERROR: {S.LISTENING},
    S.ENDING: {S.ENDED},
    S.ENDED: set(),
}


class IllegalTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(
        self,
        on_change: Callable[[S, S | None, str | None], None],
        initial: S = S.CONNECTING,
    ) -> None:
        self.state = initial
        self._on_change = on_change
        self.history: list[tuple[S, str | None]] = [(initial, None)]

    def can(self, new: S) -> bool:
        if self.state in (S.ENDING, S.ENDED):
            return new in ALLOWED[self.state]
        return new in ALLOWED[self.state] or new in _ALWAYS

    def to(self, new: S, reason: str | None = None) -> None:
        if new == self.state:
            return
        if not self.can(new):
            raise IllegalTransition(f"{self.state.value} → {new.value} ({reason})")
        prev = self.state
        self.state = new
        self.history.append((new, reason))
        self._on_change(new, prev, reason)

    def try_to(self, new: S, reason: str | None = None) -> bool:
        try:
            self.to(new, reason)
        except IllegalTransition as exc:
            log.warning("ignored illegal transition: %s", exc)
            return False
        return True
