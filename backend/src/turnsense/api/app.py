"""FastAPI service: capabilities, sessions (simulation + LiveKit tokens), event WebSocket,
scripted evaluations, trial history, comparison and exports. See docs/api-contract.md."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from turnsense import __version__
from turnsense.api.managers import (
    ApiError,
    EvaluationManager,
    SessionManager,
    live_providers,
    load_catalog,
)
from turnsense.compare import (
    compare_trials,
    comparison_csv,
    comparison_markdown,
    trial_events_csv,
    trial_json,
    trials_summary_csv,
    trials_summary_json,
)
from turnsense.config import Settings, get_settings
from turnsense.detectors import detector_availability
from turnsense.events import (
    PROTOCOL_VERSION,
    CreateEvaluationRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    EvaluationStatus,
    PolicyConfig,
)
from turnsense.runtime import SIM_PROVIDERS, policy_limits
from turnsense.trial import TrialRecord, TrialStore

log = logging.getLogger(__name__)


class TrialExportRequest(BaseModel):
    trial_ids: list[str] = Field(min_length=1, max_length=1000)
    format: Literal["json", "csv"] = "json"


class CompareRequest(BaseModel):
    trial_ids: list[str] | None = Field(None, max_length=2000)
    evaluation_id: str | None = None
    group_by: list[str] = Field(default_factory=lambda: ["detector_mode"], max_length=4)


def _download(body: str | bytes, filename: str, media_type: str) -> Response:
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = TrialStore(settings.trials_dir)
    catalog = load_catalog(settings)
    sessions = SessionManager(settings, store, catalog)
    evaluations = EvaluationManager(settings, store, catalog)
    started = time.monotonic()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        sessions.start_background()
        yield
        await sessions.shutdown()

    app = FastAPI(title="turnsense", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.sessions = sessions
    app.state.evaluations = evaluations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=["Content-Disposition"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    # ------------------------------------------------------------- service info

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "server_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "uptime_s": round(time.monotonic() - started, 1),
        }

    @app.get("/api/capabilities")
    async def capabilities() -> dict[str, Any]:
        detectors = await asyncio.to_thread(detector_availability, settings.checkpoint_dir)
        livekit_reason = (
            None
            if settings.livekit_configured
            else "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET not configured"
        )
        live = settings.livekit_configured
        return {
            "server_version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "detectors": [
                {
                    "mode": d.mode.value,
                    "label": d.label,
                    "model_version": d.model_version,
                    "available": d.available,
                    "unavailable_reason": d.unavailable_reason,
                    "probabilistic": d.probabilistic,
                    **({"details": d.extra} if d.extra else {}),
                }
                for d in detectors
            ],
            "scenarios": [
                {
                    "id": s.id,
                    "label": s.label,
                    "description": s.description,
                    "trial_type": s.trial_type.value,
                    "labelled": s.labelled,
                    "transports": list(s.transports),
                }
                for s in catalog.values()
            ],
            "transports": {
                "simulation": {"available": True, "reason": None},
                "livekit": {"available": live, "reason": livekit_reason},
            },
            "policy_defaults": PolicyConfig().model_dump(),
            "policy_limits": policy_limits(),
            "privacy": {
                "raw_audio_retention": False,
                "transcript_display_default": True,
                "transcript_export_default": False,
            },
            "runtime_profile": settings.runtime_profile if live else "simulation",
            "providers": (live_providers(settings) if live else SIM_PROVIDERS).model_dump(),
            "min_percentile_samples": settings.min_percentile_samples,
            "max_evaluation_repeats": 50,
        }

    # ------------------------------------------------------------- sessions

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest) -> CreateSessionResponse:
        return await sessions.create(req)

    @app.post("/api/sessions/{session_id}/interrupt", status_code=202)
    async def interrupt(session_id: str) -> dict[str, bool]:
        if not await sessions.interrupt(session_id):
            raise ApiError(409, "Nothing to interrupt: the agent is not generating or speaking")
        return {"accepted": True}

    @app.post("/api/sessions/{session_id}/end", status_code=202)
    async def end_session(session_id: str) -> dict[str, bool]:
        await sessions.end(session_id)
        return {"accepted": True}

    @app.websocket("/ws/sessions/{session_id}")
    async def session_events(ws: WebSocket, session_id: str) -> None:
        ms = sessions.sessions.get(session_id)
        if ms is None or ms.sim is None or ms.sim.bus is None or (ms.finished and ms.task is None):
            await ws.close(code=4404)
            return
        await ws.accept()
        bus = ms.sim.bus
        sub, replay = bus.subscribe()
        sessions.ensure_started(ms)

        async def _drain_client() -> None:  # we only expect pings; detect disconnects
            with contextlib.suppress(WebSocketDisconnect):
                while True:
                    await ws.receive_text()

        reader = asyncio.create_task(_drain_client())
        try:
            for env in replay:
                await ws.send_text(env.model_dump_json())
            while True:
                get = asyncio.create_task(sub.queue.get())
                done, _ = await asyncio.wait({get, reader}, return_when=asyncio.FIRST_COMPLETED)
                if reader in done:
                    get.cancel()
                    break
                env = get.result()
                if env is None:
                    break
                await ws.send_text(env.model_dump_json())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            reader.cancel()
            bus.unsubscribe(sub)
            with contextlib.suppress(Exception):
                await ws.close()

    # ------------------------------------------------------------- evaluations

    @app.post("/api/evaluations", status_code=202, response_model=EvaluationStatus)
    async def create_evaluation(req: CreateEvaluationRequest) -> EvaluationStatus:
        return evaluations.start(req)

    @app.get("/api/evaluations/{evaluation_id}", response_model=EvaluationStatus)
    async def get_evaluation(evaluation_id: str) -> EvaluationStatus:
        return evaluations.get(evaluation_id)

    @app.post("/api/evaluations/{evaluation_id}/stop", response_model=EvaluationStatus)
    async def stop_evaluation(evaluation_id: str) -> EvaluationStatus:
        return evaluations.stop(evaluation_id)

    # ------------------------------------------------------------- trials

    def _load(trial_id: str) -> TrialRecord:
        rec = store.get(trial_id)
        if rec is None:
            raise HTTPException(404, "Unknown trial")
        return rec

    def _load_many(ids: list[str]) -> list[TrialRecord]:
        out = []
        for tid in dict.fromkeys(ids):
            rec = store.get(tid)
            if rec is None:
                raise HTTPException(404, f"Unknown trial {tid}")
            out.append(rec)
        return out

    @app.get("/api/trials")
    async def list_trials(
        detector_mode: str | None = None,
        scenario_id: str | None = None,
        trial_type: str | None = None,
        evaluation_id: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        total, items = await asyncio.to_thread(
            store.list,
            detector_mode=detector_mode,
            scenario_id=scenario_id,
            trial_type=trial_type,
            evaluation_id=evaluation_id,
            limit=limit,
            offset=offset,
        )
        return {"total": total, "items": items}

    @app.get("/api/trials/{trial_id}")
    async def get_trial(trial_id: str) -> dict[str, Any]:
        return (await asyncio.to_thread(_load, trial_id)).detail()

    @app.get("/api/trials/{trial_id}/export")
    async def export_trial(trial_id: str, format: Literal["json", "csv"] = "json") -> Response:
        rec = await asyncio.to_thread(_load, trial_id)
        if format == "csv":
            return _download(trial_events_csv(rec), f"{trial_id}_events.csv", "text/csv")
        return JSONResponse(
            trial_json(rec),
            headers={"Content-Disposition": f'attachment; filename="{trial_id}.json"'},
        )

    @app.post("/api/trials/export")
    async def export_trials(req: TrialExportRequest) -> Response:
        recs = await asyncio.to_thread(_load_many, req.trial_ids)
        if req.format == "csv":
            return _download(trials_summary_csv(recs), "trials_summary.csv", "text/csv")
        return JSONResponse(
            trials_summary_json(recs),
            headers={"Content-Disposition": 'attachment; filename="trials_summary.json"'},
        )

    @app.post("/api/compare")
    async def compare(
        req: CompareRequest, format: Literal["json", "csv", "md"] = "json"
    ) -> Response:
        ids = list(req.trial_ids or [])
        if req.evaluation_id:
            ids += evaluations.get(req.evaluation_id).trial_ids
        if not ids:
            raise ApiError(400, "Provide trial_ids or an evaluation_id")
        recs = await asyncio.to_thread(_load_many, ids)
        result = await asyncio.to_thread(
            compare_trials, recs, req.group_by, settings.min_percentile_samples
        )
        if format == "csv":
            return _download(comparison_csv(result), "comparison.csv", "text/csv")
        if format == "md":
            return _download(comparison_markdown(result), "comparison.md", "text/markdown")
        return JSONResponse(result)

    @app.get("/", include_in_schema=False)
    async def root() -> PlainTextResponse:
        return PlainTextResponse("turnsense API — see /docs and docs/api-contract.md")

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="turnsense API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(
        "turnsense.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
