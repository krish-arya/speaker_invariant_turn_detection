"""E0: VAD + fixed silence timeout — the transparent deployable baseline (goal.md §6)."""

from __future__ import annotations

from turnsense.detectors.base import DetectorInput, Prediction, TurnDetector
from turnsense.events import DetectorMode, PolicyConfig, TurnEventType

SILENCE_BASELINE_VERSION = "silence-baseline-0.1.0"


class SilenceBaselineDetector(TurnDetector):
    """Deterministic: confidence is always None.

    * speech present                         → CONTINUE
    * silence ≥ ``possible_end_ms``          → POSSIBLE_END
    * silence ≥ ``silence_timeout_ms``       → TURN_COMPLETE
    """

    mode = DetectorMode.SILENCE_BASELINE
    model_version = SILENCE_BASELINE_VERSION
    probabilistic = False

    def __init__(self, policy: PolicyConfig) -> None:
        super().__init__(policy)
        self._user_spoke = False

    def reset(self) -> None:
        super().reset()
        self._user_spoke = False

    def new_turn(self) -> None:
        super().new_turn()
        self._user_spoke = False

    def predict(self, inp: DetectorInput) -> Prediction:
        overlap = self.speech_overlap_event(inp)
        if overlap is not None:
            return overlap
        vad = inp.vad
        if vad.is_speech:
            self._user_spoke = True
            return Prediction(TurnEventType.CONTINUE, None, reason_codes=("speech_present",))
        if not self._user_spoke:
            return Prediction(TurnEventType.CONTINUE, None, reason_codes=("no_user_speech_yet",))
        silence = vad.silence_run_ms
        if silence >= self.policy.silence_timeout_ms:
            return Prediction(TurnEventType.TURN_COMPLETE, None, reason_codes=("silence_timeout",))
        if silence >= self.policy.possible_end_ms:
            return Prediction(TurnEventType.POSSIBLE_END, None, reason_codes=("pause",))
        return Prediction(TurnEventType.CONTINUE, None, reason_codes=("short_gap",))
