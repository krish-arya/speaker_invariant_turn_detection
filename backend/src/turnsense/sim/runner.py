"""Runs a scripted scenario through the real pipeline (VAD → detector → policy → orchestrator)
with simulated providers, in real time (interactive demo) or virtual time (batch evaluation).
"""

from __future__ import annotations

import asyncio
import logging

from turnsense import __version__
from turnsense.adapters.mock import MockLLM, MockSTT, MockTTS, TimedNullSink
from turnsense.audio.frames import AudioFrame, FrameBuffer
from turnsense.audio.vad import EnergyVAD
from turnsense.detectors.base import TurnDetector
from turnsense.events import EnvelopeKind, HelloPayload, SessionInfo, SessionState
from turnsense.orchestrator.session import Orchestrator, OrchestratorOptions
from turnsense.sim.scenarios import BuiltScenario
from turnsense.tracing import EventBus, SessionClock, Tracer
from turnsense.trial import TrialRecord, TrialRecorder, TrialStore, build_summary, now_iso

log = logging.getLogger(__name__)

IDLE_STATES = {SessionState.LISTENING, SessionState.ENDED, SessionState.ERROR}


class SimulatedSession:
    """Owns one simulated trial. ``setup`` and ``run`` must be called on the same event loop."""

    def __init__(
        self,
        info: SessionInfo,
        scenario: BuiltScenario,
        detector: TurnDetector,
        store: TrialStore | None,
        *,
        min_samples: int = 20,
        evaluation_id: str | None = None,
        options: OrchestratorOptions | None = None,
        max_tail_ms: float = 15_000.0,
    ) -> None:
        self.info = info
        self.scenario = scenario
        self.detector = detector
        self.store = store
        self.min_samples = min_samples
        self.evaluation_id = evaluation_id
        self.options = options or OrchestratorOptions()
        self.max_tail_ms = max_tail_ms
        self.record: TrialRecord | None = None
        self._stop = asyncio.Event()
        self._started = False
        self.bus: EventBus | None = None
        self.orchestrator: Orchestrator | None = None

    def setup(self) -> None:
        cfg = self.info.effective_config
        self.clock = SessionClock()
        self.bus = EventBus(self.info.session_id, self.clock)
        self.tracer = Tracer(self.bus, self.info.trial_id, cfg.detector_mode, self.min_samples)
        self.recorder = TrialRecorder(keep_transcripts=cfg.transcript_export_enabled)
        self.bus.add_listener(self.recorder)
        self.orchestrator = Orchestrator(
            info=self.info,
            detector=self.detector,
            vad=EnergyVAD(cfg.vad),
            stt=MockSTT(self.scenario.words),
            llm=MockLLM(self.scenario.seed, sentences=self.scenario.llm_sentences),
            tts=MockTTS(self.scenario.seed),
            sink=TimedNullSink(),
            bus=self.bus,
            tracer=self.tracer,
            options=self.options,
        )

    @property
    def started(self) -> bool:
        return self._started

    def request_stop(self) -> None:
        self._stop.set()

    async def interrupt(self) -> bool:
        return await self.orchestrator.interrupt() if self.orchestrator else False

    async def run(self) -> TrialRecord:
        if self.orchestrator is None:
            self.setup()
        assert self.orchestrator is not None and self.bus is not None
        self._started = True
        orch, bus = self.orchestrator, self.bus
        self.clock.reset()  # origin = first audio sample
        bus.emit(
            EnvelopeKind.HELLO,
            HelloPayload(server_version=__version__, session=self.info),
        )
        created_at = self.clock.wall_iso(0)
        status = "completed"
        try:
            await orch.start()
            frames = FrameBuffer().push(self.scenario.audio)
            for frame in frames:
                if self._stop.is_set():
                    status = "aborted"
                    break
                await self.clock.sleep_until(frame.end_ms)
                orch.process_frame(_received(frame, self.clock.now_ms()))
            # let an in-flight response finish, as a real user would wait (bounded)
            deadline = self.clock.now_ms() + self.max_tail_ms
            while (
                status == "completed"
                and orch.state not in IDLE_STATES
                and self.clock.now_ms() < deadline
                and not self._stop.is_set()
            ):
                await asyncio.sleep(0.05)
        except Exception:
            log.exception("simulated session %s failed", self.info.session_id)
            status = "error"
        await orch.end("scenario_complete" if status == "completed" else status)

        summary, resolved, samples = build_summary(
            trial_id=self.info.trial_id,
            session_id=self.info.session_id,
            trial_type=self.info.trial_type,
            config=self.info.effective_config,
            orchestrator=orch,
            script=self.scenario.annotations if status == "completed" else None,
            status=status,  # type: ignore[arg-type]
            min_samples=self.min_samples,
        )
        if status != "completed":
            summary.notes.append("Trial did not complete; task metrics withheld.")
        bus.emit(EnvelopeKind.TRIAL_SUMMARY, summary)
        orch.mark_ended()
        bus.close()

        cfg = self.info.effective_config
        notes = list(summary.notes)
        if self.recorder.truncated:
            notes.append("Event log truncated at the per-trial limit.")
        self.record = TrialRecord(
            trial_id=self.info.trial_id,
            session_id=self.info.session_id,
            evaluation_id=self.evaluation_id,
            created_at=created_at,
            ended_at=self.clock.wall_iso(),
            trial_type=self.info.trial_type,
            transport=self.info.transport,
            scenario_id=self.info.scenario_id,
            trial_label=self.info.trial_label,
            detector_mode=cfg.detector_mode.value,
            model_version=cfg.model_version,
            decision_policy_version=cfg.decision_policy_version,
            status=status,  # type: ignore[arg-type]
            labelled=summary.labelled,
            metrics=summary.metrics,
            counts=summary.counts,
            effective_config=cfg,
            annotations=resolved,
            tolerance_ms=summary.tolerance_ms,
            latency_samples=samples,
            notes=notes,
            events=self.recorder.events,
        )
        if self.store is not None:
            self.store.save(self.record)
        return self.record


def _received(frame: AudioFrame, now_ms: float) -> AudioFrame:
    return AudioFrame(
        frame.seq, frame.start_ms, frame.end_ms, frame.samples, frame.sample_rate, now_ms
    )


__all__ = ["SimulatedSession", "now_iso"]
