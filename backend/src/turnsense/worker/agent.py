"""LiveKit agent worker: one orchestrator per room (room name = session ID).

Pipeline per session::

    participant mic ──AudioStream(16 kHz, 20 ms)──► FrameBuffer ──bounded queue──► Orchestrator
    Orchestrator ──► PlaybackController ──► LiveKitSink (AudioSource, flushed on barge-in)
    EventBus ──ordered publisher──► data channel topic "turnsense.events"
    API ──RoomService.send_data──► topic "turnsense.control" {"action": "interrupt" | "end"}

The session's frozen configuration is the JSON the API wrote when it issued the room token;
the worker never accepts configuration from the browser.

Run:  turnsense-worker dev     (reads LIVEKIT_* and provider keys from backend/.env)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re

import numpy as np

from turnsense import __version__
from turnsense.adapters.base import PlaybackSink
from turnsense.adapters.factory import ProviderConfigError, build_live_providers
from turnsense.audio.frames import (
    AudioFrame,
    FrameBuffer,
    float_to_pcm16,
    pcm16_to_float,
    resample_linear,
)
from turnsense.audio.vad import EnergyVAD, create_vad
from turnsense.config import Settings, get_settings
from turnsense.detectors import create_detector
from turnsense.events import (
    EnvelopeKind,
    ErrorPayload,
    ErrorSource,
    HelloPayload,
    SessionInfo,
    Transport,
    TrialType,
)
from turnsense.livekit_service import CONTROL_TOPIC, EVENTS_TOPIC
from turnsense.orchestrator.session import Orchestrator, OrchestratorOptions
from turnsense.tracing import EventBus, SessionClock, Tracer, iso, utc_now
from turnsense.trial import TrialRecord, TrialRecorder, TrialStore, build_summary

log = logging.getLogger("turnsense.worker")
_SESSION_ID = re.compile(r"^ses_[0-9a-f]{16}$")


class LiveKitSink(PlaybackSink):
    """Feeds agent audio into a LiveKit AudioSource in 10 ms frames."""

    def __init__(self, source) -> None:  # type: ignore[no-untyped-def]
        self._source = source
        self._rate = source.sample_rate

    async def play(self, samples: np.ndarray, sample_rate: int) -> None:
        from livekit import rtc

        if sample_rate != self._rate:
            samples = resample_linear(samples, sample_rate, self._rate)
        step = self._rate // 100
        for i in range(0, samples.size, step):
            chunk = samples[i : i + step]
            await self._source.capture_frame(
                rtc.AudioFrame(float_to_pcm16(chunk), self._rate, 1, chunk.size)
            )

    def flush(self) -> None:
        self._source.clear_queue()

    async def drain(self) -> None:
        await self._source.wait_for_playout()


class FrameQueue:
    """Bounded hand-off from the transport reader to the processor; drops oldest on overload."""

    def __init__(self, max_frames: int = 50) -> None:
        self._q: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=max_frames)
        self.dropped = 0

    def put(self, frame: AudioFrame) -> None:
        while True:
            try:
                self._q.put_nowait(frame)
                return
            except asyncio.QueueFull:
                self._q.get_nowait()
                self.dropped += 1

    async def get(self) -> AudioFrame:
        return await self._q.get()


def load_session_info(settings: Settings, session_id: str) -> SessionInfo | None:
    if not _SESSION_ID.match(session_id):
        return None
    path = settings.sessions_dir / f"{session_id}.json"
    if not path.is_file():
        return None
    return SessionInfo.model_validate_json(path.read_text(encoding="utf-8"))


async def entrypoint(ctx) -> None:  # type: ignore[no-untyped-def]  # livekit.agents.JobContext
    from livekit import rtc
    from livekit.agents import AutoSubscribe

    settings = get_settings()
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    room = ctx.room
    info = load_session_info(settings, room.name)
    if info is None:
        log.warning("room %s has no session created by the API; leaving", room.name)
        ctx.shutdown(reason="unknown session")
        return
    cfg = info.effective_config
    clock = SessionClock()
    bus = EventBus(info.session_id, clock)

    # ordered data-channel publisher (sync bus listener → single async sender)
    outbox: asyncio.Queue[str | None] = asyncio.Queue(maxsize=5000)

    def _enqueue(env) -> None:  # type: ignore[no-untyped-def]
        with contextlib.suppress(asyncio.QueueFull):
            outbox.put_nowait(env.model_dump_json())

    async def _publisher() -> None:
        while (msg := await outbox.get()) is not None:
            try:
                await room.local_participant.publish_data(msg, reliable=True, topic=EVENTS_TOPIC)
            except Exception:
                log.exception("publish_data failed")

    bus.add_listener(_enqueue)
    publisher = asyncio.create_task(_publisher())
    bus.emit(EnvelopeKind.HELLO, HelloPayload(server_version=__version__, session=info))

    try:
        providers = build_live_providers(settings)
    except (ProviderConfigError, ImportError) as exc:
        bus.emit(
            EnvelopeKind.ERROR,
            ErrorPayload(
                source=ErrorSource.WORKER,
                code="providers_unavailable",
                message=str(exc),
                recoverable=False,
            ),
        )
        await asyncio.sleep(0.5)
        outbox.put_nowait(None)
        await publisher
        ctx.shutdown(reason="providers unavailable")
        return

    notes: list[str] = []
    try:
        vad = create_vad(cfg.vad)
    except (FileNotFoundError, ImportError) as exc:
        notes.append(f"Silero VAD unavailable ({exc}); energy VAD used instead.")
        vad = EnergyVAD(cfg.vad)
    detector = create_detector(cfg.detector_mode, cfg.policy, settings.checkpoint_dir)
    tracer = Tracer(bus, info.trial_id, cfg.detector_mode, settings.min_percentile_samples)
    recorder = TrialRecorder(keep_transcripts=cfg.transcript_export_enabled)
    bus.add_listener(recorder)

    source = rtc.AudioSource(providers.tts.sample_rate, 1, queue_size_ms=400)
    agent_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
    await room.local_participant.publish_track(
        agent_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    )
    final_wait = (
        settings.stt_final_wait_ms
        if settings.runtime_profile == "hosted"
        else max(settings.stt_final_wait_ms, 2500)
    )
    orch = Orchestrator(
        info=info,
        detector=detector,
        vad=vad,
        stt=providers.stt,
        llm=providers.llm,
        tts=providers.tts,
        sink=LiveKitSink(source),
        bus=bus,
        tracer=tracer,
        options=OrchestratorOptions(
            system_prompt=settings.system_prompt,
            stt_final_wait_ms=final_wait,
            llm_timeout_s=settings.llm_timeout_s,
            tts_timeout_s=settings.tts_timeout_s,
            speculative_prefill=cfg.speculative_prefill,
        ),
    )

    done = asyncio.Event()
    end_reason = {"value": "participant_left"}
    participant = await ctx.wait_for_participant()

    @room.on("participant_disconnected")
    def _on_left(p) -> None:  # type: ignore[no-untyped-def]
        if p.identity == participant.identity:
            done.set()

    @room.on("reconnecting")
    def _on_reconnecting() -> None:
        orch.set_reconnecting(True)

    @room.on("reconnected")
    def _on_reconnected() -> None:
        orch.set_reconnecting(False)

    @room.on("data_received")
    def _on_data(packet) -> None:  # type: ignore[no-untyped-def]
        # control messages come from the API via RoomService (no participant), never from browsers
        if packet.topic != CONTROL_TOPIC or packet.participant is not None:
            return
        try:
            action = json.loads(bytes(packet.data)).get("action")
        except ValueError:
            return
        if action == "interrupt":
            asyncio.create_task(orch.interrupt())
        elif action == "end":
            end_reason["value"] = "user_ended"
            done.set()

    await orch.start()
    fb = FrameBuffer(gap_tolerance_ms=200)
    frames = FrameQueue()

    async def _reader(track) -> None:  # type: ignore[no-untyped-def]
        stream = rtc.AudioStream(track, sample_rate=16_000, num_channels=1, frame_size_ms=20)
        align: float | None = None
        async for ev in stream:
            now = clock.now_ms()
            pcm = pcm16_to_float(np.frombuffer(bytes(ev.frame.data), dtype=np.int16))
            dur = pcm.size / 16.0
            if align is None:
                align = now - dur
            for f in fb.push(pcm, capture_ms=now - dur - align, received_ms=now):
                frames.put(f)

    async def _processor() -> None:
        while True:
            frame = await frames.get()
            orch.process_frame(frame)

    def _mic_track():  # type: ignore[no-untyped-def]
        for pub in participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO and pub.track is not None:
                return pub.track
        return None

    track = _mic_track()
    if track is None:
        got: asyncio.Future = asyncio.get_running_loop().create_future()

        @room.on("track_subscribed")
        def _on_track(t, pub, p) -> None:  # type: ignore[no-untyped-def]
            if (
                p.identity == participant.identity
                and t.kind == rtc.TrackKind.KIND_AUDIO
                and not got.done()
            ):
                got.set_result(t)

        track = await got
    tasks = [asyncio.create_task(_reader(track)), asyncio.create_task(_processor())]
    await done.wait()

    for t in tasks:
        t.cancel()
    await orch.end(end_reason["value"])
    summary, _, samples = build_summary(
        trial_id=info.trial_id,
        session_id=info.session_id,
        trial_type=TrialType.LIVE,
        config=cfg,
        orchestrator=orch,
        script=None,
        status="completed",
        dropped_frames=fb.dropped_frames + frames.dropped,
        min_samples=settings.min_percentile_samples,
    )
    summary.notes.extend(notes)
    bus.emit(EnvelopeKind.TRIAL_SUMMARY, summary)
    orch.mark_ended()
    bus.close()
    outbox.put_nowait(None)
    with contextlib.suppress(Exception):
        await asyncio.wait_for(publisher, timeout=3)

    TrialStore(settings.trials_dir).save(
        TrialRecord(
            trial_id=info.trial_id,
            session_id=info.session_id,
            created_at=info.started_at,
            ended_at=iso(utc_now()),
            trial_type=TrialType.LIVE,
            transport=Transport.LIVEKIT,
            scenario_id=info.scenario_id,
            trial_label=info.trial_label,
            detector_mode=cfg.detector_mode.value,
            model_version=cfg.model_version,
            decision_policy_version=cfg.decision_policy_version,
            status="completed",
            labelled=False,
            metrics=summary.metrics,
            counts=summary.counts,
            effective_config=cfg,
            latency_samples=samples,
            notes=summary.notes,
            events=recorder.events,
        )
    )
    with contextlib.suppress(Exception):
        (settings.sessions_dir / f"{info.session_id}.json").unlink()
    ctx.shutdown(reason=end_reason["value"])


def _export_env(settings: Settings) -> None:
    """LiveKit Agents reads LIVEKIT_* from the process environment; mirror backend/.env there."""
    if settings.livekit_url:
        os.environ.setdefault("LIVEKIT_URL", settings.livekit_url)
    if settings.livekit_api_key:
        os.environ.setdefault("LIVEKIT_API_KEY", settings.livekit_api_key.get_secret_value())
    if settings.livekit_api_secret:
        os.environ.setdefault("LIVEKIT_API_SECRET", settings.livekit_api_secret.get_secret_value())


def main() -> None:
    from livekit.agents import WorkerOptions, cli

    settings = get_settings()
    _export_env(settings)
    if not settings.livekit_configured:
        raise SystemExit("Set LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in backend/.env")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
