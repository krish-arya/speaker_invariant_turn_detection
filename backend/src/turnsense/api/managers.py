"""Session and evaluation lifecycle for the API process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field

from turnsense.config import Settings
from turnsense.detectors import create_detector, detector_availability
from turnsense.events import (
    CreateEvaluationRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    DetectorMode,
    EvaluationStatus,
    LiveKitJoinInfo,
    ProviderInfo,
    SessionInfo,
    Transport,
    TrialType,
)
from turnsense.orchestrator.session import OrchestratorOptions
from turnsense.runtime import SIM_PROVIDERS, build_effective_config, build_policy
from turnsense.sim.replay import scenario_catalog
from turnsense.sim.runner import SimulatedSession
from turnsense.sim.scenarios import BuiltScenario, ScenarioSpec
from turnsense.sim.vtime import run_in_virtual_time
from turnsense.tracing import iso, new_id, utc_now
from turnsense.trial import TrialStore

log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def live_providers(settings: Settings) -> ProviderInfo:
    if settings.runtime_profile == "local":
        return ProviderInfo(
            stt=f"faster_whisper_{settings.whisper_model}",
            llm=f"ollama_{settings.ollama_model}",
            tts="kokoro_82m",
        )
    return ProviderInfo(
        stt=f"deepgram_{settings.stt_model.replace('-', '_')}",
        llm=f"groq_{settings.llm_model.replace('-', '_').replace('.', '_')}",
        tts=f"deepgram_{settings.tts_voice.replace('-', '_')}",
    )


def _model_version(settings: Settings, mode: DetectorMode) -> str:
    for a in detector_availability(settings.checkpoint_dir):
        if a.mode == mode:
            if not a.available:
                raise ApiError(
                    409, f"Detector '{mode.value}' is unavailable: {a.unavailable_reason}"
                )
            return a.model_version
    raise ApiError(409, f"Unknown detector '{mode.value}'")


@dataclass
class ManagedSession:
    info: SessionInfo
    created: float = field(default_factory=time.monotonic)
    sim: SimulatedSession | None = None
    task: asyncio.Task[object] | None = None
    finished: bool = False


def load_catalog(settings: Settings) -> dict[str, ScenarioSpec]:
    return scenario_catalog(settings.replay_manifest, settings.replay_split)


def _build(catalog: dict[str, ScenarioSpec], scenario_id: str, seed: int) -> BuiltScenario:
    spec = catalog[scenario_id]
    if spec.build is None:
        raise ApiError(400, f"Scenario '{scenario_id}' has no scripted user track")
    return spec.build(seed)


class SessionManager:
    def __init__(
        self, settings: Settings, store: TrialStore, catalog: dict[str, ScenarioSpec]
    ) -> None:
        self.settings = settings
        self.store = store
        self.catalog = catalog
        self.sessions: dict[str, ManagedSession] = {}
        self._janitor: asyncio.Task[None] | None = None

    def start_background(self) -> None:
        self._janitor = asyncio.create_task(self._janitor_loop())

    async def shutdown(self) -> None:
        if self._janitor:
            self._janitor.cancel()
        for ms in list(self.sessions.values()):
            if ms.sim and ms.task and not ms.task.done():
                ms.sim.request_stop()
        tasks = [ms.task for ms in self.sessions.values() if ms.task and not ms.task.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=5)

    def active_count(self) -> int:
        return sum(1 for s in self.sessions.values() if not s.finished)

    async def create(self, req: CreateSessionRequest) -> CreateSessionResponse:
        spec = self.catalog.get(req.scenario_id)
        if spec is None:
            raise ApiError(400, f"Unknown scenario '{req.scenario_id}'")
        if req.transport.value not in spec.transports:
            raise ApiError(
                400,
                f"Scenario '{req.scenario_id}' does not support transport '{req.transport.value}'",
            )
        if self.active_count() >= self.settings.max_concurrent_sessions:
            raise ApiError(429, "Too many active sessions; end one first")
        if req.transport == Transport.LIVEKIT and not self.settings.livekit_configured:
            raise ApiError(503, "LiveKit is not configured on this server")
        try:
            policy = build_policy(req.policy_overrides)
        except ValueError as exc:
            raise ApiError(422, str(exc)) from exc
        model_version = _model_version(self.settings, req.detector_mode)
        seed = req.seed if req.seed is not None else int(time.time()) % 100_000
        live = req.transport == Transport.LIVEKIT
        cfg = build_effective_config(
            detector_mode=req.detector_mode,
            model_version=model_version,
            policy=policy,
            transport=req.transport,
            scenario_id=req.scenario_id,
            runtime_profile=self.settings.runtime_profile if live else "simulation",
            providers=live_providers(self.settings) if live else SIM_PROVIDERS,
            vad_kind=self.settings.live_vad if live else "energy",
            transcript_enabled=req.transcript_enabled,
            transcript_export_enabled=req.transcript_export_enabled,
            seed=None if live else seed,
            speculative_prefill=req.speculative_prefill,
        )
        session_id, trial_id = new_id("ses"), new_id("tri")
        info = SessionInfo(
            session_id=session_id,
            trial_id=trial_id,
            transport=req.transport,
            trial_type=spec.trial_type,
            scenario_id=req.scenario_id,
            trial_label=req.trial_label,
            effective_config=cfg,
            started_at=iso(utc_now()),
        )
        managed = ManagedSession(info=info)
        if live:
            self.settings.sessions_dir.mkdir(parents=True, exist_ok=True)
            (self.settings.sessions_dir / f"{session_id}.json").write_text(
                info.model_dump_json(), encoding="utf-8"
            )
            from turnsense.livekit_service import mint_room_token

            token, expires = mint_room_token(self.settings, session_id, f"user-{session_id[-8:]}")
            self.sessions[session_id] = managed
            return CreateSessionResponse(
                session_id=session_id,
                trial_id=trial_id,
                effective_config=cfg,
                events_url=None,
                livekit=LiveKitJoinInfo(
                    url=self.settings.livekit_url or "",
                    room=session_id,
                    token=token,
                    expires_at=expires,
                ),
            )

        detector = create_detector(req.detector_mode, policy, self.settings.checkpoint_dir)
        scenario = await asyncio.to_thread(_build, self.catalog, req.scenario_id, seed)
        sim = SimulatedSession(
            info,
            scenario,
            detector,
            self.store,
            min_samples=self.settings.min_percentile_samples,
            options=OrchestratorOptions(
                system_prompt=self.settings.system_prompt,
                emit_audio_levels=True,
                speculative_prefill=cfg.speculative_prefill,
            ),
        )
        sim.setup()
        managed.sim = sim
        self.sessions[session_id] = managed
        return CreateSessionResponse(
            session_id=session_id,
            trial_id=trial_id,
            effective_config=cfg,
            events_url=f"/ws/sessions/{session_id}",
            livekit=None,
        )

    def get(self, session_id: str) -> ManagedSession:
        ms = self.sessions.get(session_id)
        if ms is None:
            raise ApiError(404, "Unknown session")
        return ms

    def ensure_started(self, ms: ManagedSession) -> None:
        if ms.sim is None or ms.task is not None:
            return

        async def _run() -> object:
            try:
                return await ms.sim.run()  # type: ignore[union-attr]
            finally:
                ms.finished = True

        ms.task = asyncio.create_task(_run(), name=f"sim-{ms.info.session_id}")

    async def interrupt(self, session_id: str) -> bool:
        ms = self.get(session_id)
        if ms.finished:
            raise ApiError(409, "Session has ended")
        if ms.sim is not None:
            return await ms.sim.interrupt()
        from turnsense.livekit_service import send_control

        await send_control(self.settings, session_id, "interrupt")
        return True

    async def end(self, session_id: str) -> None:
        ms = self.get(session_id)
        if ms.finished:
            return
        if ms.sim is not None:
            if ms.task is None:  # never started: nothing to record
                ms.finished = True
                if ms.sim.bus:
                    ms.sim.bus.close()
                return
            ms.sim.request_stop()
            return
        from turnsense.livekit_service import send_control

        with contextlib.suppress(Exception):
            await send_control(self.settings, session_id, "end")
        ms.finished = True

    async def _janitor_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            now = time.monotonic()
            for sid, ms in list(self.sessions.items()):
                if ms.sim is not None and ms.task is None and now - ms.created > 60:
                    ms.finished = True  # nobody connected: discard without a record
                if ms.finished and now - ms.created > 600:
                    self.sessions.pop(sid, None)


@dataclass
class EvaluationJob:
    status: EvaluationStatus
    request: CreateEvaluationRequest
    stop: threading.Event = field(default_factory=threading.Event)


class EvaluationManager:
    """Runs scripted trials server-side in virtual time, one worker thread per evaluation."""

    MAX_RUNNING = 2

    def __init__(
        self, settings: Settings, store: TrialStore, catalog: dict[str, ScenarioSpec]
    ) -> None:
        self.settings = settings
        self.store = store
        self.catalog = catalog
        self.jobs: dict[str, EvaluationJob] = {}
        self._lock = threading.Lock()

    def start(self, req: CreateEvaluationRequest) -> EvaluationStatus:
        running = sum(1 for j in self.jobs.values() if j.status.status in ("queued", "running"))
        if running >= self.MAX_RUNNING:
            raise ApiError(429, "Too many evaluations running; wait or stop one")
        for sid in req.scenario_ids:
            spec = self.catalog.get(sid)
            if spec is None or spec.build is None:
                raise ApiError(400, f"Scenario '{sid}' cannot be run as a scripted evaluation")
        try:
            build_policy(req.policy_overrides)
        except ValueError as exc:
            raise ApiError(422, str(exc)) from exc
        for mode in req.detector_modes:
            _model_version(self.settings, mode)
        modes = list(dict.fromkeys(req.detector_modes))
        scenarios = list(dict.fromkeys(req.scenario_ids))
        status = EvaluationStatus(
            evaluation_id=new_id("eva"),
            status="queued",
            total=len(modes) * len(scenarios) * req.repeats,
            completed=0,
            trial_ids=[],
            created_at=iso(utc_now()),
        )
        job = EvaluationJob(
            status=status,
            request=req.model_copy(update={"detector_modes": modes, "scenario_ids": scenarios}),
        )
        self.jobs[status.evaluation_id] = job
        threading.Thread(
            target=self._run, args=(job,), daemon=True, name=status.evaluation_id
        ).start()
        return status

    def get(self, evaluation_id: str) -> EvaluationStatus:
        job = self.jobs.get(evaluation_id)
        if job is None:
            raise ApiError(404, "Unknown evaluation")
        return job.status

    def stop(self, evaluation_id: str) -> EvaluationStatus:
        job = self.jobs.get(evaluation_id)
        if job is None:
            raise ApiError(404, "Unknown evaluation")
        job.stop.set()
        return job.status

    def _run(self, job: EvaluationJob) -> None:
        req, st = job.request, job.status
        st.status = "running"
        policy = build_policy(req.policy_overrides)
        try:
            for repeat in range(req.repeats):
                seed = req.seed + repeat
                for scenario_id in req.scenario_ids:
                    # identical audio for every mode
                    scenario = _build(self.catalog, scenario_id, seed)
                    for mode in req.detector_modes:
                        if job.stop.is_set():
                            st.status = "stopped"
                            return
                        trial_id = self._run_one(
                            req, st.evaluation_id, mode, scenario, seed, policy
                        )
                        with self._lock:
                            st.trial_ids.append(trial_id)
                            st.completed += 1
            st.status = "completed"
        except Exception as exc:
            log.exception("evaluation %s failed", st.evaluation_id)
            st.status = "error"
            st.error = repr(exc)

    def _run_one(self, req, evaluation_id, mode, scenario, seed, policy) -> str:  # type: ignore[no-untyped-def]
        spec = self.catalog[scenario.scenario_id]
        cfg = build_effective_config(
            detector_mode=mode,
            model_version=_model_version(self.settings, mode),
            policy=policy,
            transport=Transport.SIMULATION,
            scenario_id=scenario.scenario_id,
            runtime_profile="simulation",
            providers=SIM_PROVIDERS,
            vad_kind="energy",
            transcript_enabled=False,
            transcript_export_enabled=False,
            seed=seed,
            speculative_prefill=req.speculative_prefill,
        )
        info = SessionInfo(
            session_id=new_id("ses"),
            trial_id=new_id("tri"),
            transport=Transport.SIMULATION,
            trial_type=spec.trial_type
            if spec.trial_type != TrialType.LIVE
            else TrialType.SIMULATED,
            scenario_id=scenario.scenario_id,
            trial_label=req.trial_label,
            effective_config=cfg,
            started_at=iso(utc_now()),
        )
        detector = create_detector(mode, policy, self.settings.checkpoint_dir)
        sim = SimulatedSession(
            info,
            scenario,
            detector,
            self.store,
            min_samples=self.settings.min_percentile_samples,
            evaluation_id=evaluation_id,
            options=OrchestratorOptions(
                system_prompt=self.settings.system_prompt,
                speculative_prefill=cfg.speculative_prefill,
            ),
        )
        run_in_virtual_time(sim.run)
        return info.trial_id
