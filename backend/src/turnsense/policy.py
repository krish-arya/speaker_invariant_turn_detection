"""Decision policy: turns detector predictions into committed decisions (architecture.md §5.4).

The same class is used by the live orchestrator and by offline evaluation, so a metric
computed offline describes exactly the behaviour the orchestrator would exhibit.

Rules (all thresholds come from ``PolicyConfig`` and are recorded per trial):

* While the agent is active (generating or speaking):
  ``BARGE_IN`` prediction → commit ``BARGE_IN`` immediately;
  ``BACKCHANNEL`` prediction → commit ``BACKCHANNEL_IGNORED`` (logged, no state change).
* Otherwise, once the user has produced at least ``min_speech_ms`` of speech this turn:
  commit ``TURN_COMPLETE`` when the detector predicts it with silence ≥
  ``min_silence_guard_ms`` — immediately for deterministic detectors, after
  ``consecutive_frames`` consecutive predictions for probabilistic ones.
"""

from __future__ import annotations

from dataclasses import dataclass

from turnsense.audio.vad import VadResult
from turnsense.detectors.base import DetectorOutput
from turnsense.events import DecisionType, PolicyConfig, TurnEventType


@dataclass(frozen=True, slots=True)
class CommittedDecision:
    decision: DecisionType
    source: DetectorOutput
    last_user_frame_ms: float  # end of the newest frame the decision used
    confidence: float | None
    reason_codes: tuple[str, ...]


class DecisionPolicy:
    def __init__(self, policy: PolicyConfig, frame_ms: float = 20.0) -> None:
        self.config = policy
        self.frame_ms = frame_ms
        self.reset()

    def reset(self) -> None:
        self._consecutive = 0
        self._turn_speech_ms = 0.0

    @property
    def turn_speech_ms(self) -> float:
        return self._turn_speech_ms

    def new_turn(self) -> None:
        self._consecutive = 0
        self._turn_speech_ms = 0.0

    def update(
        self, out: DetectorOutput, vad: VadResult, *, agent_active: bool
    ) -> CommittedDecision | None:
        cfg = self.config
        if agent_active:
            self._consecutive = 0
            if out.event_type == TurnEventType.BARGE_IN:
                return self._commit(DecisionType.BARGE_IN, out, ("barge_in_predicted",))
            if out.event_type == TurnEventType.BACKCHANNEL:
                return self._commit(
                    DecisionType.BACKCHANNEL_IGNORED, out, ("backchannel_predicted",)
                )
            return None

        if vad.is_speech:
            self._turn_speech_ms += self.frame_ms
        if out.event_type != TurnEventType.TURN_COMPLETE:
            self._consecutive = 0
            return None
        self._consecutive += 1
        if self._turn_speech_ms < cfg.min_speech_ms:
            return None
        if vad.silence_run_ms < cfg.min_silence_guard_ms:
            return None
        probabilistic = out.p_turn_end is not None
        # the max-silence fallback is a deterministic bound and commits at once
        if (
            probabilistic
            and self._consecutive < cfg.consecutive_frames
            and "max_silence_reached" not in out.reason_codes
        ):
            return None
        reasons = (*out.reason_codes, f"consecutive={self._consecutive}")
        return self._commit(DecisionType.TURN_COMPLETE, out, reasons)

    def _commit(
        self, decision: DecisionType, out: DetectorOutput, reasons: tuple[str, ...]
    ) -> CommittedDecision:
        if decision == DecisionType.TURN_COMPLETE:
            self.new_turn()
        return CommittedDecision(
            decision=decision,
            source=out,
            last_user_frame_ms=out.frame_end_ms,
            confidence=out.confidence,
            reason_codes=reasons,
        )
