"""The orchestrator: single owner of session state (architecture.md §4.4).

Per 20 ms frame: VAD → STT push → detector prediction → decision policy → state machine.
Committed TURN_COMPLETE starts a response *generation* (LLM → clause chunker → TTS →
playback), tagged with a generation ID. A committed BARGE_IN cancels the active generation:
playback is flushed synchronously, late chunks are rejected by the playback controller, and the
assistant history keeps only what was actually played.

The orchestrator contains no model-specific logic; detectors are swapped via the registry.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np

from turnsense.adapters.base import (
    ChatMessage,
    LLMAdapter,
    PlaybackSink,
    ProviderError,
    STTAdapter,
    SttEvent,
    TTSAdapter,
)
from turnsense.adapters.chunker import ClauseChunker
from turnsense.audio.frames import AudioFrame, rms_dbfs
from turnsense.audio.vad import VadResult, VoiceActivityDetector
from turnsense.detectors.base import DetectorInput, TranscriptHint, TurnDetector
from turnsense.events import (
    AudioLevelPayload,
    DecisionType,
    EnvelopeKind,
    ErrorPayload,
    ErrorSource,
    LatencyMetric,
    PlaybackPayload,
    SessionInfo,
    SessionState,
    SessionStatusPayload,
    TraceStatus,
    TranscriptPayload,
    TurnDecisionPayload,
    TurnEvent,
    TurnEventType,
    VadPayload,
)
from turnsense.orchestrator.playback import PlaybackController
from turnsense.orchestrator.state import StateMachine
from turnsense.policy import CommittedDecision, DecisionPolicy
from turnsense.tracing import EventBus, Tracer

log = logging.getLogger(__name__)
S = SessionState


@dataclass
class OrchestratorOptions:
    system_prompt: str = "You are a concise, friendly voice assistant."
    stt_final_wait_ms: float = 600.0
    llm_timeout_s: float = 15.0
    tts_timeout_s: float = 15.0
    emit_audio_levels: bool = False
    history_messages: int = 12
    summary_interval_s: float = 1.0
    error_recovery_ms: float = 500.0
    interrupted_idle_ms: float = 300.0
    # Speculative prefill: start the LLM during POSSIBLE_END once the partial transcript has
    # been stable this long; reuse it only if the committed turn's text is unchanged.
    speculative_prefill: bool = False
    prefill_stable_ms: float = 150.0


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


@dataclass
class Speculation:
    """An LLM stream started before the turn was committed, tied to one transcript text."""

    text: str
    started_ms: float
    tokens: list[str] = field(default_factory=list)
    first_token_ms: float | None = None
    done: bool = False
    error: BaseException | None = None
    updated: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None

    async def replay(self) -> AsyncIterator[str]:
        i = 0
        while True:
            while i < len(self.tokens):
                yield self.tokens[i]
                i += 1
            if self.error is not None:
                raise self.error
            if self.done:
                return
            self.updated.clear()
            await self.updated.wait()


@dataclass
class Generation:
    id: int
    decided_ms: float
    last_user_frame_ms: float
    user_text: str = ""
    task: asyncio.Task[None] | None = None
    cancelled: bool = False
    cancel_reason: str | None = None
    failed: bool = False
    first_audio_ms: float | None = None
    delivered: list[str] = field(default_factory=list)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    done: bool = False
    ended_ms: float | None = None  # completed, cancelled or failed


@dataclass
class DecisionRecord:
    decision: DecisionType
    frame_end_ms: float  # audio-timeline time of the deciding frame
    committed_ms: float  # session clock when committed
    generation_id: int | None
    manual: bool = False


@dataclass
class AgentInterval:
    generation_id: int
    start_ms: float
    end_ms: float | None = None
    cancelled_by: str | None = None


@dataclass
class SessionCounters:
    frames: int = 0
    responses_started: int = 0
    responses_cancelled: int = 0
    errors: int = 0
    unanswered_turns: int = 0


class Orchestrator:
    def __init__(
        self,
        *,
        info: SessionInfo,
        detector: TurnDetector,
        vad: VoiceActivityDetector,
        stt: STTAdapter,
        llm: LLMAdapter,
        tts: TTSAdapter,
        sink: PlaybackSink,
        bus: EventBus,
        tracer: Tracer,
        options: OrchestratorOptions | None = None,
    ) -> None:
        self.info = info
        self.cfg = info.effective_config
        self.detector = detector
        self.vad = vad
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.bus = bus
        self.clock = bus.clock
        self.tracer = tracer
        self.opts = options or OrchestratorOptions()
        self.policy = DecisionPolicy(self.cfg.policy, self.cfg.frame_ms)
        self.sm = StateMachine(self._on_state)
        self.playback = PlaybackController(
            sink,
            on_first_audio=self._on_first_audio,
            on_clause=self._on_clause,
            on_drained=self._on_drained,
            on_stale=self._on_stale,
            on_chunk=self._on_agent_chunk if self.opts.emit_audio_levels else None,
            on_error=self._on_playback_error,
        )

        self.history: list[ChatMessage] = []
        self.generations: list[Generation] = []
        self.decisions: list[DecisionRecord] = []
        self.agent_intervals: list[AgentInterval] = []
        self.counters = SessionCounters()
        self._gen: Generation | None = None
        self._gen_counter = 0
        self._pred_seq = 0
        self._last_pred: TurnEventType | None = None
        self._last_vad: VadResult | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._summary_task: asyncio.Task[None] | None = None
        # user transcript for the turn being built
        self._finals: list[str] = []
        self._partial = ""
        self._segment = 1
        self._revision = 0
        self._text_changed_ms = 0.0
        self._final_event = asyncio.Event()
        self._started = False
        self._spec: Speculation | None = None

    # ------------------------------------------------------------------ lifecycle

    @property
    def state(self) -> SessionState:
        return self.sm.state

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.bus.emit(
            EnvelopeKind.SESSION_STATUS,
            SessionStatusPayload(state=S.CONNECTING, previous_state=None, reason="session_created"),
        )
        try:
            await self.stt.start(self._on_stt, self._on_stt_error)
        except ProviderError as exc:
            self._report_error(exc.source, exc.code, str(exc), exc.recoverable)
        self.playback.start()
        self._summary_task = asyncio.create_task(self._summary_loop(), name="latency-summary")
        self.sm.to(S.LISTENING, "room_joined")

    async def end(self, reason: str = "user_ended") -> None:
        if self.sm.state in (S.ENDING, S.ENDED):
            return
        self.sm.to(S.ENDING, reason)
        self._drop_speculation("session_end")
        if self._gen is not None:
            gen = self._cancel_sync(self._gen, "session_end")
            await self._await_cancel(gen, None)
        for task in list(self._tasks):
            task.cancel()
        if self._summary_task:
            self._summary_task.cancel()
        await self.playback.stop()
        with contextlib.suppress(Exception):
            await self.stt.close()
        with contextlib.suppress(Exception):
            await self.tts.close()
        self.tracer.emit_summary()

    def mark_ended(self) -> None:
        self.sm.try_to(S.ENDED, "closed")

    def set_reconnecting(self, reconnecting: bool) -> None:
        """Transport state from the worker; a gap is reported, never interpolated."""
        if reconnecting and self.sm.state not in (S.ENDING, S.ENDED):
            if self._gen is not None:
                self._cancel_sync(self._gen, "transport_lost")
            self.sm.try_to(S.RECONNECTING, "transport_reconnecting")
        elif not reconnecting and self.sm.state == S.RECONNECTING:
            self.detector.new_turn()
            self.policy.new_turn()
            self.sm.try_to(S.LISTENING, "transport_reconnected")

    # ------------------------------------------------------------------ per frame

    def process_frame(self, frame: AudioFrame) -> None:
        if self.sm.state in (S.CONNECTING, S.ENDING, S.ENDED, S.RECONNECTING):
            return
        self.counters.frames += 1
        now = self.clock.now_ms()
        vad = self.vad.process(frame)
        self._last_vad = vad
        if vad.changed:
            self.bus.emit(
                EnvelopeKind.VAD,
                VadPayload(
                    speaking=vad.is_speech, probability=vad.probability, frame_end_ms=frame.end_ms
                ),
            )
            self.tracer.point(
                "vad_speech_start" if vad.is_speech else "vad_speech_end",
                detail={"frame_seq": frame.seq, "frame_end_ms": frame.end_ms},
            )
        try:
            self.stt.push_frame(frame)
        except Exception as exc:  # adapter bug: report, keep turn detection running
            self._report_error(ErrorSource.STT, "push_failed", str(exc), True)

        agent_active = self.sm.state in (S.GENERATING, S.AGENT_SPEAKING)
        try:
            out = self.detector.process(
                DetectorInput(frame, vad, agent_speaking=agent_active, transcript=self._hint(now))
            )
        except Exception as exc:
            self.tracer.frame_processed(None)
            self._report_error(ErrorSource.DETECTOR, "inference_failed", str(exc), True)
            return
        self.tracer.frame_processed(out.inference_ms)

        if out.event_type != self._last_pred:
            self._last_pred = out.event_type
            self._pred_seq += 1
            ev = TurnEvent(
                session_id=self.info.session_id,
                sequence=self._pred_seq,
                event_type=out.event_type,
                detector_mode=self.cfg.detector_mode,
                model_version=self.cfg.model_version,
                event_end_ms=out.frame_end_ms,
                inference_ms=round(out.inference_ms, 4),
                confidence=None if out.confidence is None else round(out.confidence, 4),
                decision_policy_version=self.cfg.decision_policy_version,
                reason_codes=list(out.reason_codes),
            )
            self.bus.emit(EnvelopeKind.TURN_PREDICTION, ev, generation_id=self._gen_id())
            self.tracer.point(
                "turn_prediction",
                duration_ms=out.inference_ms,
                emit=False,
                detail={"event_type": out.event_type.value, "sequence": self._pred_seq},
            )

        decision = self.policy.update(out, vad, agent_active=agent_active)
        if decision is not None:  # a committed decision owns this frame's transition
            self._handle_decision(decision, frame, vad, now)
        else:
            self._update_listening_state(vad, out.event_type, agent_active)
        if self.opts.speculative_prefill:
            self._maybe_speculate(now)

        if self.opts.emit_audio_levels and frame.seq % 4 == 0:
            self.bus.emit(
                EnvelopeKind.AUDIO_LEVEL,
                AudioLevelPayload(source="user", rms_dbfs=round(rms_dbfs(frame.samples), 1)),
            )

    def _update_listening_state(
        self, vad: VadResult, pred: TurnEventType, agent_active: bool
    ) -> None:
        st = self.sm.state
        if agent_active or st in (S.ERROR,):
            return
        pol = self.cfg.policy
        if vad.is_speech:
            if st in (S.LISTENING, S.INTERRUPTED):
                self.sm.to(S.USER_SPEAKING, "speech_detected")
            elif st == S.POSSIBLE_END:
                self._drop_speculation("speech_resumed")
                self.sm.to(S.USER_SPEAKING, "speech_resumed")
            return
        if st in (S.USER_SPEAKING, S.POSSIBLE_END):
            too_short = self.policy.turn_speech_ms < pol.min_speech_ms
            if too_short and vad.silence_run_ms >= pol.possible_end_ms:
                self.detector.new_turn()
                self.policy.new_turn()
                self.sm.to(S.LISTENING, "speech_too_short")
            elif st == S.USER_SPEAKING and pred in (
                TurnEventType.POSSIBLE_END,
                TurnEventType.TURN_COMPLETE,
            ):
                self.sm.to(S.POSSIBLE_END, "pause")
        elif st == S.INTERRUPTED and vad.silence_run_ms >= self.opts.interrupted_idle_ms:
            self.sm.to(S.LISTENING, "no_speech_after_interrupt")

    # ------------------------------------------------------------------ decisions

    def _handle_decision(
        self, d: CommittedDecision, frame: AudioFrame, vad: VadResult, now: float
    ) -> None:
        last_frame_ms = frame.received_ms if frame.received_ms is not None else frame.end_ms
        gen_id = self._gen_id()
        self.bus.emit(
            EnvelopeKind.TURN_DECISION,
            TurnDecisionPayload(
                decision=d.decision,
                source_event_sequence=self._pred_seq,
                last_user_frame_ms=round(last_frame_ms, 3),
                committed_ms=round(now, 3),
                confidence=None if d.confidence is None else round(d.confidence, 4),
                reason_codes=list(d.reason_codes),
            ),
            generation_id=gen_id,
        )
        self.tracer.point(
            "turn_decision",
            emit=False,
            generation_id=gen_id,
            detail={"decision": d.decision.value, "frame_seq": frame.seq},
        )
        self.decisions.append(DecisionRecord(d.decision, frame.end_ms, now, gen_id))

        if d.decision == DecisionType.TURN_COMPLETE:
            self.detector.new_turn()
            try:
                self.stt.finalize()
            except Exception as exc:
                self._report_error(ErrorSource.STT, "finalize_failed", str(exc), True)
            if self.sm.state == S.USER_SPEAKING:
                self.sm.to(S.POSSIBLE_END, "pause")
            self._start_generation(last_frame_ms, now)
        elif d.decision == DecisionType.BARGE_IN:
            self._barge_in(now, "barge_in", user_speaking=vad.is_speech)
        else:  # BACKCHANNEL_IGNORED: logged; the agent keeps the floor
            self.tracer.point("backchannel_ignored", generation_id=gen_id)

    def _barge_in(self, now: float, reason: str, user_speaking: bool) -> bool:
        gen = self._gen
        if gen is None:
            return False
        self.tracer.point(
            "barge_in_detected", t_ms=now, generation_id=gen.id, detail={"reason": reason}
        )
        self.detector.new_turn()
        self.policy.new_turn()
        self._cancel_sync(gen, reason)
        self._spawn(self._await_cancel(gen, now))
        self.sm.to(S.INTERRUPTED, reason)
        if user_speaking:
            self.sm.to(S.USER_SPEAKING, "user_continues")
        return True

    async def interrupt(self, reason: str = "manual_interrupt") -> bool:
        """Manual 'Interrupt agent': the same cancellation path as a detected barge-in."""
        if self._gen is None or self.sm.state not in (S.GENERATING, S.AGENT_SPEAKING):
            return False
        now = self.clock.now_ms()
        self.bus.emit(
            EnvelopeKind.TURN_DECISION,
            TurnDecisionPayload(
                decision=DecisionType.BARGE_IN,
                source_event_sequence=None,
                last_user_frame_ms=round(now, 3),
                committed_ms=round(now, 3),
                confidence=None,
                reason_codes=[reason],
            ),
            generation_id=self._gen.id,
        )
        self.decisions.append(
            DecisionRecord(DecisionType.BARGE_IN, now, now, self._gen.id, manual=True)
        )
        speaking = bool(self._last_vad and self._last_vad.is_speech)
        return self._barge_in(now, reason, user_speaking=speaking)

    # ------------------------------------------------------------------ generations

    def _gen_id(self) -> int | None:
        return self._gen.id if self._gen else None

    # speculative prefill -------------------------------------------------------

    def _maybe_speculate(self, now: float) -> None:
        if self.sm.state != S.POSSIBLE_END or self._gen is not None:
            return
        text = self._user_text()
        if not text:
            return
        if self._spec is not None:
            if _norm(self._spec.text) == _norm(text):
                return
            self._drop_speculation("transcript_changed")
        if now - self._text_changed_ms < self.opts.prefill_stable_ms:
            return
        spec = Speculation(text=text, started_ms=now)
        messages = [
            ChatMessage("system", self.opts.system_prompt),
            *self.history[-self.opts.history_messages :],
            ChatMessage("user", text),
        ]
        spec.task = asyncio.create_task(self._speculate(spec, messages), name="speculation")
        self._spec = spec
        self.tracer.point("llm_request", provider=self.llm.name, detail={"speculative": True})

    async def _speculate(self, spec: Speculation, messages: list[ChatMessage]) -> None:
        try:
            async with (
                asyncio.timeout(self.opts.llm_timeout_s),
                contextlib.aclosing(self.llm.stream(messages)) as stream,
            ):
                async for delta in stream:
                    if spec.first_token_ms is None:
                        spec.first_token_ms = self.clock.now_ms()
                    spec.tokens.append(delta)
                    spec.updated.set()
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            spec.error = ProviderError(ErrorSource.LLM, "timeout", "LLM response timed out")
        except ProviderError as exc:
            spec.error = exc
        except Exception as exc:
            spec.error = ProviderError(ErrorSource.LLM, "speculation_failed", repr(exc))
        finally:
            spec.done = True
            spec.updated.set()

    def _drop_speculation(self, reason: str) -> None:
        spec, self._spec = self._spec, None
        if spec is None:
            return
        if spec.task is not None and not spec.task.done():
            spec.task.cancel()
        self.tracer.point(
            "speculation_discarded", status=TraceStatus.CANCELLED, detail={"reason": reason}
        )

    def _claim_speculation(self, user_text: str) -> Speculation | None:
        spec = self._spec
        if spec is None:
            return None
        if _norm(spec.text) != _norm(user_text):
            self._drop_speculation("final_transcript_differs")
            return None
        self._spec = None
        return spec

    def _start_generation(self, last_user_frame_ms: float, decided_ms: float) -> None:
        if self._gen is not None:  # cannot normally happen: a turn completes only when idle
            self._cancel_sync(self._gen, "superseded")
        self._gen_counter += 1
        gen = Generation(self._gen_counter, decided_ms, last_user_frame_ms)
        self._gen = gen
        self.generations.append(gen)
        self.counters.responses_started += 1
        self.playback.begin(gen.id)
        self.sm.to(S.GENERATING, "turn_complete")
        gen.task = asyncio.create_task(self._run_generation(gen), name=f"generation-{gen.id}")

    async def _run_generation(self, gen: Generation) -> None:
        tts_task: asyncio.Task[None] | None = None
        spec: Speculation | None = None
        try:
            user_text = await self._await_user_text()
            if not user_text:
                self.counters.unanswered_turns += 1
                self._report_error(
                    ErrorSource.STT,
                    "no_transcript",
                    "No transcript was available for the completed turn, so no reply was generated.",
                    True,
                    generation_id=gen.id,
                )
                self._finish(gen, S.LISTENING, "no_transcript")
                return
            gen.user_text = user_text
            messages = [
                ChatMessage("system", self.opts.system_prompt),
                *self.history[-self.opts.history_messages :],
                ChatMessage("user", user_text),
            ]
            spec = self._claim_speculation(user_text)
            if spec is not None:
                t_req = spec.started_ms
                source = spec.replay()
                self.tracer.point(
                    "speculation_used",
                    generation_id=gen.id,
                    detail={"head_start_ms": round(self.clock.now_ms() - spec.started_ms, 1)},
                )
            else:
                t_req = self.clock.now_ms()
                source = self.llm.stream(messages)
                self.tracer.point("llm_request", provider=self.llm.name, generation_id=gen.id)
            clauses: asyncio.Queue[str | None] = asyncio.Queue()
            tts_task = asyncio.create_task(self._tts_loop(gen, clauses), name=f"tts-{gen.id}")
            chunker = ClauseChunker()
            first = True
            try:
                async with (
                    asyncio.timeout(self.opts.llm_timeout_s),
                    contextlib.aclosing(source) as stream,
                ):
                    async for delta in stream:
                        if first:
                            first = False
                            arrived = spec.first_token_ms if spec is not None else None
                            dt = (arrived if arrived is not None else self.clock.now_ms()) - t_req
                            self.tracer.point(
                                "llm_first_token",
                                duration_ms=dt,
                                provider=self.llm.name,
                                generation_id=gen.id,
                            )
                            self.tracer.sample(
                                LatencyMetric.LLM_FIRST_TOKEN, dt, generation_id=gen.id
                            )
                        for clause in chunker.push(delta):
                            clauses.put_nowait(clause)
            except TimeoutError as exc:
                raise ProviderError(ErrorSource.LLM, "timeout", "LLM response timed out") from exc
            for clause in chunker.flush():
                clauses.put_nowait(clause)
            self.tracer.point(
                "llm_complete",
                duration_ms=self.clock.now_ms() - t_req,
                provider=self.llm.name,
                generation_id=gen.id,
            )
            clauses.put_nowait(None)
            await tts_task
            self.playback.end_of_generation(gen.id)
            await gen.drained.wait()
            self._complete(gen)
        except asyncio.CancelledError:
            if tts_task:
                tts_task.cancel()
            if spec is not None and spec.task is not None:
                spec.task.cancel()
            raise
        except ProviderError as exc:
            if tts_task:
                tts_task.cancel()
            self._fail(gen, exc.source, exc.code, str(exc), exc.recoverable)
        except Exception as exc:  # unexpected adapter failure: bounded, reported, recoverable
            if tts_task:
                tts_task.cancel()
            log.exception("generation %s failed", gen.id)
            self._fail(gen, ErrorSource.WORKER, "generation_failed", repr(exc), True)

    async def _tts_loop(self, gen: Generation, clauses: asyncio.Queue[str | None]) -> None:
        first_overall = True
        while (clause := await clauses.get()) is not None:
            t0 = self.clock.now_ms()
            self.tracer.point(
                "tts_request",
                provider=self.tts.name,
                generation_id=gen.id,
                detail={"chars": len(clause)},
            )
            first_chunk = True
            try:
                async with (
                    asyncio.timeout(self.opts.tts_timeout_s),
                    contextlib.aclosing(self.tts.synthesize(clause)) as stream,
                ):
                    async for chunk in stream:
                        if first_chunk:
                            dt = self.clock.now_ms() - t0
                            self.tracer.point(
                                "tts_first_audio",
                                duration_ms=dt,
                                provider=self.tts.name,
                                generation_id=gen.id,
                            )
                            if first_overall:
                                self.tracer.sample(
                                    LatencyMetric.TTS_FIRST_AUDIO, dt, generation_id=gen.id
                                )
                                first_overall = False
                        self.playback.put(
                            gen.id, chunk, self.tts.sample_rate, clause if first_chunk else None
                        )
                        first_chunk = False
            except TimeoutError as exc:
                raise ProviderError(
                    ErrorSource.TTS, "timeout", "Speech synthesis timed out"
                ) from exc

    async def _await_user_text(self) -> str:
        """Wait (bounded) for STT to finalise the turn, then use the best text available."""
        deadline = self.clock.now_ms() + self.opts.stt_final_wait_ms
        while True:
            if self._finals and not self._partial:
                return self._user_text()
            remaining = deadline - self.clock.now_ms()
            if remaining <= 0:
                return self._user_text()
            self._final_event.clear()
            try:
                await asyncio.wait_for(self._final_event.wait(), remaining / 1000)
            except TimeoutError:
                return self._user_text()

    # playback callbacks ------------------------------------------------------

    def _lookup(self, gen_id: int) -> Generation | None:
        gen = self._gen
        return gen if gen is not None and gen.id == gen_id and not gen.cancelled else None

    def _on_first_audio(self, gen_id: int) -> None:
        gen = self._lookup(gen_id)
        if gen is None:
            return
        now = self.clock.now_ms()
        gen.first_audio_ms = now
        latency = now - gen.last_user_frame_ms
        self.tracer.point("playback_start", generation_id=gen.id, duration_ms=latency)
        self.tracer.sample(LatencyMetric.RESPONSE_START, latency, generation_id=gen.id)
        self.bus.emit(EnvelopeKind.PLAYBACK, PlaybackPayload(action="start"), generation_id=gen.id)
        self.agent_intervals.append(AgentInterval(gen.id, now))
        # the user's turn is now answered: commit it to history and start a new segment
        self.history.append(ChatMessage("user", gen.user_text))
        self._emit_user_transcript(final=True)
        self._finals, self._partial = [], ""
        self._segment += 1
        self._revision = 0
        if self.sm.state == S.GENERATING:
            self.sm.to(S.AGENT_SPEAKING, "first_audio")

    def _on_clause(self, gen_id: int, clause: str) -> None:
        gen = self._lookup(gen_id)
        if gen is None:
            return
        gen.delivered.append(clause)
        self._emit_assistant_transcript(gen, final=False)

    def _on_drained(self, gen_id: int) -> None:
        gen = self._lookup(gen_id)
        if gen is not None:
            gen.drained.set()

    def _on_stale(self, gen_id: int) -> None:
        self.tracer.point(
            "stale_chunk_rejected", generation_id=gen_id, status=TraceStatus.CANCELLED
        )

    def _on_agent_chunk(self, gen_id: int, samples: np.ndarray) -> None:
        self.bus.emit(
            EnvelopeKind.AUDIO_LEVEL,
            AudioLevelPayload(source="agent", rms_dbfs=round(rms_dbfs(samples), 1)),
            generation_id=gen_id,
        )

    def _on_playback_error(self, gen_id: int, exc: BaseException) -> None:
        gen = self._lookup(gen_id)
        if gen is not None:
            if gen.task:
                gen.task.cancel()
            self._fail(gen, ErrorSource.TRANSPORT, "playback_failed", repr(exc), True)

    # completion / cancellation / failure -------------------------------------

    def _close_interval(self, gen: Generation, cancelled_by: str | None) -> None:
        for interval in reversed(self.agent_intervals):
            if interval.generation_id == gen.id and interval.end_ms is None:
                interval.end_ms = self.clock.now_ms()
                interval.cancelled_by = cancelled_by
                return

    def _complete(self, gen: Generation) -> None:
        gen.done = True
        if gen.first_audio_ms is not None:
            self.history.append(ChatMessage("assistant", " ".join(gen.delivered)))
            self._emit_assistant_transcript(gen, final=True)
            self.tracer.point(
                "playback_end",
                generation_id=gen.id,
                duration_ms=self.clock.now_ms() - gen.first_audio_ms,
            )
            self.bus.emit(
                EnvelopeKind.PLAYBACK, PlaybackPayload(action="end"), generation_id=gen.id
            )
            self._close_interval(gen, None)
        self._finish(
            gen, S.LISTENING, "playback_completed" if gen.first_audio_ms else "empty_response"
        )

    def _finish(self, gen: Generation, next_state: SessionState, reason: str) -> None:
        if gen.ended_ms is None:
            gen.ended_ms = self.clock.now_ms()
        if self._gen is gen:
            self._gen = None
        if self.sm.state in (S.GENERATING, S.AGENT_SPEAKING):
            self.sm.to(next_state, reason)

    def _cancel_sync(self, gen: Generation, reason: str) -> Generation:
        """Synchronous half of cancellation: nothing of ``gen`` may play after this returns."""
        if gen.cancelled:
            return gen
        gen.cancelled = True
        gen.cancel_reason = reason
        gen.ended_ms = self.clock.now_ms()
        self.tracer.point("cancel_requested", generation_id=gen.id, detail={"reason": reason})
        dropped = self.playback.cancel(gen.id)
        self.bus.emit(
            EnvelopeKind.PLAYBACK,
            PlaybackPayload(action="flush", reason=f"{reason}; dropped_chunks={dropped}"),
            generation_id=gen.id,
        )
        if gen.first_audio_ms is not None:
            self._close_interval(gen, reason)
            if gen.delivered:  # keep only what the user actually heard
                self.history.append(ChatMessage("assistant", " ".join(gen.delivered) + " …"))
            self._emit_assistant_transcript(gen, final=True)
        if self._gen is gen:
            self._gen = None
        self.counters.responses_cancelled += 1
        if gen.task and not gen.task.done():
            gen.task.cancel()
        return gen

    async def _await_cancel(self, gen: Generation, decided_ms: float | None) -> None:
        t0 = self.clock.now_ms()
        if gen.task is not None:
            await asyncio.wait({gen.task}, timeout=2.0)
        now = self.clock.now_ms()
        status = TraceStatus.OK if gen.task is None or gen.task.done() else TraceStatus.TIMEOUT
        self.tracer.point(
            "cancel_acknowledged", generation_id=gen.id, duration_ms=now - t0, status=status
        )
        if decided_ms is not None:
            self.tracer.sample(
                LatencyMetric.BARGE_IN_CANCEL, now - decided_ms, status, generation_id=gen.id
            )

    def _fail(
        self, gen: Generation, source: ErrorSource, code: str, message: str, recoverable: bool
    ) -> None:
        gen.failed = True
        if gen.ended_ms is None:
            gen.ended_ms = self.clock.now_ms()
        self.playback.cancel(gen.id)
        self._close_interval(gen, f"error:{code}")
        if gen.first_audio_ms is None:
            self.tracer.sample(
                LatencyMetric.RESPONSE_START, None, TraceStatus.ERROR, generation_id=gen.id
            )
        self.tracer.point(
            "provider_error",
            status=TraceStatus.ERROR,
            generation_id=gen.id,
            detail={"source": source.value, "code": code},
        )
        self._report_error(source, code, message, recoverable, generation_id=gen.id)
        if self._gen is gen:
            self._gen = None
        if self.sm.state not in (S.ENDING, S.ENDED):
            self.sm.to(S.ERROR, f"{source.value}:{code}")
            self._spawn(self._recover())

    async def _recover(self) -> None:
        await asyncio.sleep(self.opts.error_recovery_ms / 1000)
        if self.sm.state == S.ERROR:
            self.detector.new_turn()
            self.policy.new_turn()
            self.sm.to(S.LISTENING, "recovered")

    # ------------------------------------------------------------------ STT

    def _on_stt(self, ev: SttEvent) -> None:
        now = self.clock.now_ms()
        if ev.is_final:
            self._finals.append(ev.text)
            self._partial = ""
            latency = now - ev.end_ms if ev.end_ms is not None else None
            self.tracer.point("stt_final", provider=self.stt.name, duration_ms=latency)
            if latency is not None:
                self.tracer.sample(LatencyMetric.STT_FINAL, latency)
        else:
            self._partial = ev.text
            self.tracer.point("stt_partial", provider=self.stt.name, emit=False)
        self._text_changed_ms = now
        self._emit_user_transcript(final=False, start_ms=ev.start_ms, end_ms=ev.end_ms)
        if ev.is_final:
            self._final_event.set()

    def _on_stt_error(self, exc: ProviderError) -> None:
        self.tracer.point("provider_error", status=TraceStatus.ERROR, provider=self.stt.name)
        self._report_error(exc.source, exc.code, str(exc), exc.recoverable)

    def _user_text(self) -> str:
        return " ".join(p for p in [*self._finals, self._partial] if p).strip()

    def _hint(self, now: float) -> TranscriptHint | None:
        text = self._user_text()
        if not text:
            return None
        return TranscriptHint(text, not self._partial, self._revision, now - self._text_changed_ms)

    def _emit_user_transcript(
        self, final: bool, start_ms: float | None = None, end_ms: float | None = None
    ) -> None:
        text = self._user_text()
        if not text:
            return
        self._revision += 1
        if self.cfg.transcript_enabled:
            self.bus.emit(
                EnvelopeKind.TRANSCRIPT,
                TranscriptPayload(
                    segment_id=f"u{self._segment}",
                    role="user",
                    text=text,
                    is_final=final or not self._partial,
                    revision=self._revision,
                    start_ms=start_ms,
                    end_ms=end_ms,
                ),
            )

    def _emit_assistant_transcript(self, gen: Generation, final: bool) -> None:
        if not self.cfg.transcript_enabled or not gen.delivered:
            return
        self.bus.emit(
            EnvelopeKind.TRANSCRIPT,
            TranscriptPayload(
                segment_id=f"a{gen.id}",
                role="assistant",
                text=" ".join(gen.delivered) + (" …" if gen.cancelled else ""),
                is_final=final,
                revision=len(gen.delivered) + (1 if final else 0),
            ),
            generation_id=gen.id,
        )

    # ------------------------------------------------------------------ misc

    def _on_state(self, new: SessionState, prev: SessionState | None, reason: str | None) -> None:
        self.bus.emit(
            EnvelopeKind.SESSION_STATUS,
            SessionStatusPayload(state=new, previous_state=prev, reason=reason),
            generation_id=self._gen_id(),
        )

    def _report_error(
        self,
        source: ErrorSource,
        code: str,
        message: str,
        recoverable: bool,
        generation_id: int | None = None,
    ) -> None:
        self.counters.errors += 1
        self.bus.emit(
            EnvelopeKind.ERROR,
            ErrorPayload(source=source, code=code, message=message, recoverable=recoverable),
            generation_id=generation_id,
        )

    def _spawn(self, coro) -> None:  # type: ignore[no-untyped-def]
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _summary_loop(self) -> None:
        while True:
            await asyncio.sleep(self.opts.summary_interval_s)
            self.tracer.emit_summary()

    @property
    def stale_chunks_rejected(self) -> int:
        return self.playback.stale_rejected

    @property
    def stale_chunks_played(self) -> int:
        return self.playback.stale_played
