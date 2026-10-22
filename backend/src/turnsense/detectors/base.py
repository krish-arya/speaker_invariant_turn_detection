"""Turn detector adapter contract (architecture.md §4.7, §5.3).

Every detector — threshold rule, local model or hosted model — consumes the same
``DetectorInput`` per 20 ms frame and returns a ``DetectorOutput`` *prediction*. Committing a
state transition is the decision policy's job (turnsense.policy), not the detector's.

Barge-in and backchannel handling is shared by all variants (``speech_overlap_event``) so the
experiment isolates the one thing that differs: how turn completion is estimated.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from turnsense.audio.frames import AudioFrame
from turnsense.audio.vad import VadResult
from turnsense.events import DetectorMode, PolicyConfig, TurnEventType


@dataclass(frozen=True, slots=True)
class TranscriptHint:
    """Latest user transcript state, when STT is available. Detectors must not require it."""

    text: str
    is_final: bool
    revision: int
    stable_ms: float  # time since the text last changed


@dataclass(frozen=True, slots=True)
class DetectorInput:
    frame: AudioFrame
    vad: VadResult
    agent_speaking: bool  # agent audio is playing (barge-in context)
    transcript: TranscriptHint | None = None


@dataclass(frozen=True, slots=True)
class Prediction:
    event_type: TurnEventType
    confidence: float | None  # None for deterministic detectors — never fabricated
    p_turn_end: float | None = None  # learned detectors: P(user has yielded the turn)
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectorOutput:
    event_type: TurnEventType
    confidence: float | None
    p_turn_end: float | None
    reason_codes: tuple[str, ...]
    inference_ms: float
    frame_seq: int
    frame_end_ms: float


@dataclass
class _OverlapState:
    speech_ms: float = 0.0
    active: bool = False


class TurnDetector(ABC):
    """Base class. Subclasses implement ``predict``; ``process`` adds timing."""

    mode: DetectorMode
    model_version: str
    probabilistic: bool = False

    def __init__(self, policy: PolicyConfig) -> None:
        self.policy = policy
        self._overlap = _OverlapState()

    def process(self, inp: DetectorInput) -> DetectorOutput:
        t0 = time.perf_counter()
        pred = self.predict(inp)
        inference_ms = (time.perf_counter() - t0) * 1000.0
        return DetectorOutput(
            event_type=pred.event_type,
            confidence=pred.confidence,
            p_turn_end=pred.p_turn_end,
            reason_codes=pred.reason_codes,
            inference_ms=inference_ms,
            frame_seq=inp.frame.seq,
            frame_end_ms=inp.frame.end_ms,
        )

    @abstractmethod
    def predict(self, inp: DetectorInput) -> Prediction: ...

    def reset(self) -> None:
        """Clear all streaming state (new session)."""
        self._overlap = _OverlapState()

    def new_turn(self) -> None:
        """Called by the orchestrator after it commits TURN_COMPLETE or BARGE_IN.

        Clears per-turn bookkeeping; model context may be kept across turns.
        """
        self._overlap = _OverlapState()

    def info(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "model_version": self.model_version,
            "probabilistic": self.probabilistic,
        }

    # -- shared barge-in logic -------------------------------------------------

    def speech_overlap_event(self, inp: DetectorInput) -> Prediction | None:
        """While the agent speaks: BARGE_IN after sustained speech, BACKCHANNEL for short bursts.

        Returns None when the agent is not speaking, so callers fall through to turn logic.
        """
        if not inp.agent_speaking:
            self._overlap = _OverlapState()
            return None
        st = self._overlap
        if inp.vad.is_speech:
            st.active = True
            st.speech_ms = inp.vad.speech_run_ms
            if st.speech_ms >= self.policy.barge_in_min_speech_ms:
                return Prediction(
                    TurnEventType.BARGE_IN, None, reason_codes=("sustained_overlap_speech",)
                )
            return Prediction(TurnEventType.CONTINUE, None, reason_codes=("overlap_speech_short",))
        if st.active:  # a burst just ended without reaching the barge-in threshold
            st.active = False
            burst = st.speech_ms
            st.speech_ms = 0.0
            if burst <= self.policy.backchannel_max_ms:
                return Prediction(
                    TurnEventType.BACKCHANNEL, None, reason_codes=("short_overlap_burst",)
                )
        return Prediction(TurnEventType.CONTINUE, None, reason_codes=("agent_speaking",))


def map_turn_end_probability(
    p_end: float, vad: VadResult, policy: PolicyConfig, user_spoke: bool
) -> Prediction:
    """Shared mapping from P(turn end) to an event for every learned variant.

    Keeping this identical for ``learned_control`` and ``anti_transfer`` means differences
    between them come from the representation, not from the mapping.
    """
    if vad.is_speech:
        return Prediction(TurnEventType.CONTINUE, p_end, p_end, ("speech_present",))
    if not user_spoke:
        return Prediction(TurnEventType.CONTINUE, None, p_end, ("no_user_speech_yet",))
    silence = vad.silence_run_ms
    if silence >= policy.max_silence_ms:
        return Prediction(TurnEventType.TURN_COMPLETE, p_end, p_end, ("max_silence_reached",))
    if p_end >= policy.complete_threshold and silence >= policy.min_silence_guard_ms:
        return Prediction(TurnEventType.TURN_COMPLETE, p_end, p_end, ("high_turn_end_score",))
    if p_end >= policy.possible_end_threshold or silence >= policy.possible_end_ms:
        reasons = ("turn_end_score_rising",) if p_end >= policy.possible_end_threshold else ()
        if silence >= policy.possible_end_ms:
            reasons = (*reasons, "pause")
        return Prediction(TurnEventType.POSSIBLE_END, p_end, p_end, reasons)
    return Prediction(TurnEventType.CONTINUE, 1.0 - p_end, p_end, ("low_turn_end_score",))


@dataclass
class DetectorAvailability:
    mode: DetectorMode
    label: str
    model_version: str
    available: bool
    unavailable_reason: str | None
    probabilistic: bool
    extra: dict[str, object] = field(default_factory=dict)
