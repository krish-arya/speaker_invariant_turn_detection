import asyncio

import numpy as np
import pytest

from turnsense.adapters.base import PlaybackSink
from turnsense.adapters.chunker import ClauseChunker
from turnsense.events import SessionState as S
from turnsense.orchestrator.playback import PlaybackController
from turnsense.orchestrator.state import IllegalTransition, StateMachine


def test_state_machine_enforces_architecture_transitions():
    seen = []
    sm = StateMachine(lambda new, prev, why: seen.append((prev, new)))
    for st in (S.LISTENING, S.USER_SPEAKING, S.POSSIBLE_END, S.GENERATING, S.AGENT_SPEAKING, S.INTERRUPTED, S.USER_SPEAKING):
        sm.to(st)
    with pytest.raises(IllegalTransition):
        sm.to(S.AGENT_SPEAKING)  # cannot start speaking without generating
    sm.to(S.ENDING)
    sm.to(S.ENDED)
    with pytest.raises(IllegalTransition):
        sm.to(S.LISTENING)
    assert seen[0] == (S.CONNECTING, S.LISTENING)


def test_error_and_end_are_reachable_from_active_states():
    sm = StateMachine(lambda *a: None)
    sm.to(S.LISTENING)
    sm.to(S.ERROR)
    sm.to(S.LISTENING)
    sm.to(S.ENDING)
    assert sm.state == S.ENDING


class RecordingSink(PlaybackSink):
    def __init__(self):
        self.played = []
        self.flushes = 0

    async def play(self, samples, sample_rate):
        self.played.append(float(samples[0]))
        await asyncio.sleep(samples.size / sample_rate)

    def flush(self):
        self.flushes += 1


async def test_cancelled_generation_audio_is_never_played():
    sink = RecordingSink()
    events = []
    pc = PlaybackController(
        sink,
        on_first_audio=lambda g: events.append(("first", g)),
        on_clause=lambda g, c: events.append(("clause", g, c)),
        on_drained=lambda g: events.append(("drained", g)),
        on_stale=lambda g: events.append(("stale", g)),
    )
    pc.start()
    chunk = lambda v: np.full(1600, v, dtype=np.float32)  # noqa: E731 - 100 ms
    pc.begin(1)
    for v in (1.0, 2.0, 3.0, 4.0):
        assert pc.put(1, chunk(v), 16_000, "hello" if v == 1.0 else None)
    await asyncio.sleep(0.15)  # first chunk done, second playing
    dropped = pc.cancel(1)
    assert dropped >= 1 and sink.flushes == 1
    assert not pc.put(1, chunk(9.0), 16_000)  # late chunk from the cancelled generation
    pc.begin(2)
    pc.put(2, chunk(5.0), 16_000)
    pc.end_of_generation(2)
    await asyncio.sleep(0.3)
    await pc.stop()
    assert 9.0 not in sink.played and 3.0 not in sink.played and 4.0 not in sink.played
    assert sink.played[-1] == 5.0
    assert pc.stale_played == 0 and pc.stale_rejected >= 1
    assert ("first", 1) in events and ("first", 2) in events and ("drained", 2) in events
    assert ("drained", 1) not in events


def test_clause_chunker_emits_speakable_units():
    c = ClauseChunker(first_min_chars=10, min_chars=20)
    out = []
    for tok in ["Sure,", "I", "can", "help", "with", "that.", "The", "direct", "route", "leaves", "at", "nine,", "and", "the", "other", "one", "is", "cheaper."]:
        out += c.push(tok + " ")
    out += c.flush()
    assert out[0] == "Sure, I can help with that."
    assert " ".join(out).split() == ["Sure,", "I", "can", "help", "with", "that.", "The", "direct", "route", "leaves", "at", "nine,", "and", "the", "other", "one", "is", "cheaper."]
    assert all(o for o in out)
