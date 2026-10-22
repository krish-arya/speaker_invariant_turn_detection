"""Session clock, event bus and frame-level tracer (architecture.md §4.11).

* ``SessionClock`` reads the running event loop's monotonic clock, so the same code runs in
  real time (live sessions) and in virtual time (batch simulation, see sim/vtime.py).
  Wall-clock strings are derived as ``started_at + t_ms`` — durations are never computed
  from wall-clock values.
* ``EventBus`` stamps envelopes with a per-session ``seq`` and fans them out to bounded
  subscriber queues (slow consumers lose the oldest messages and see a ``seq`` gap rather
  than stalling the pipeline) and to synchronous listeners (trial recorder, LiveKit publisher).
* ``Tracer`` records trace points and latency samples and produces p50/p95/p99 summaries,
  keeping failed samples out of the percentiles.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from turnsense.events import (
    DetectorMode,
    Envelope,
    EnvelopeKind,
    LatencyMetric,
    LatencyPayload,
    LatencySummaryPayload,
    PercentileSummary,
    TraceEvent,
    TraceStatus,
)
from turnsense.metrics.stats import percentiles

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class SessionClock:
    """Milliseconds since the session origin on the event loop's monotonic clock."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._loop = loop or asyncio.get_running_loop()
        self._origin = self._loop.time()
        self.started_at = utc_now()

    def reset(self) -> None:
        """Move the origin to now (e.g. when audio actually starts flowing)."""
        self._origin = self._loop.time()
        self.started_at = utc_now()

    def now_ms(self) -> float:
        return (self._loop.time() - self._origin) * 1000.0

    def wall_iso(self, t_ms: float | None = None) -> str:
        t = self.now_ms() if t_ms is None else t_ms
        return iso(self.started_at + timedelta(milliseconds=t))

    async def sleep_until(self, t_ms: float) -> None:
        delay = (t_ms - self.now_ms()) / 1000.0
        if delay > 0:
            await asyncio.sleep(delay)


Listener = Callable[[Envelope], None]


class Subscription:
    def __init__(self, maxsize: int) -> None:
        self.queue: asyncio.Queue[Envelope | None] = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0

    def offer(self, env: Envelope | None) -> None:
        while True:
            try:
                self.queue.put_nowait(env)
                return
            except asyncio.QueueFull:
                with contextlib.suppress(asyncio.QueueEmpty):
                    self.queue.get_nowait()  # drop oldest; the client sees a seq gap
                    self.dropped += 1


class EventBus:
    def __init__(self, session_id: str, clock: SessionClock, replay_limit: int = 2000) -> None:
        self.session_id = session_id
        self.clock = clock
        self._seq = 0
        self._history: deque[Envelope] = deque(maxlen=replay_limit)
        self._subs: list[Subscription] = []
        self._listeners: list[Listener] = []
        self.closed = False

    def emit(
        self,
        kind: EnvelopeKind,
        payload: BaseModel | dict[str, Any],
        generation_id: int | None = None,
        t_ms: float | None = None,
    ) -> Envelope:
        self._seq += 1
        t = self.clock.now_ms() if t_ms is None else t_ms
        body = payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
        env = Envelope(
            kind=kind,
            session_id=self.session_id,
            seq=self._seq,
            t_ms=round(t, 3),
            wall_time=self.clock.wall_iso(t),
            generation_id=generation_id,
            payload=body,
        )
        if kind != EnvelopeKind.AUDIO_LEVEL:
            self._history.append(env)
        for sub in self._subs:
            sub.offer(env)
        for listener in self._listeners:
            try:
                listener(env)
            except Exception:  # a broken listener must not break the pipeline
                log.exception("event listener failed")
        return env

    def subscribe(self, maxsize: int = 2000) -> tuple[Subscription, list[Envelope]]:
        """Returns a live subscription plus the replay history emitted so far."""
        sub = Subscription(maxsize)
        self._subs.append(sub)
        if self.closed:
            sub.offer(None)
        return sub, list(self._history)

    def unsubscribe(self, sub: Subscription) -> None:
        with contextlib.suppress(ValueError):
            self._subs.remove(sub)

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def close(self) -> None:
        self.closed = True
        for sub in self._subs:
            sub.offer(None)


class LatencyStats:
    def __init__(self, min_samples: int, keep: int = 20_000) -> None:
        self.min_samples = min_samples
        self.ok: deque[float] = deque(maxlen=keep)
        self.failed = 0
        self.latest: float | None = None

    def add(self, value_ms: float | None, status: TraceStatus) -> None:
        if status == TraceStatus.OK and value_ms is not None:
            self.ok.append(value_ms)
            self.latest = value_ms
        else:
            self.failed += 1

    def summary(self) -> PercentileSummary:
        ps = percentiles(list(self.ok), (50, 95, 99), min_samples=self.min_samples)
        return PercentileSummary(
            count=len(self.ok),
            failed=self.failed,
            latest_ms=self.latest,
            p50_ms=ps[50],
            p95_ms=ps[95],
            p99_ms=ps[99],
            min_samples=self.min_samples,
        )


class Tracer:
    """Trace points + latency samples for one trial. Every point also goes out on the bus."""

    def __init__(
        self,
        bus: EventBus,
        trial_id: str,
        detector_mode: DetectorMode,
        min_samples: int = 20,
        max_traces: int = 50_000,
    ) -> None:
        self.bus = bus
        self.trial_id = trial_id
        self.detector_mode = detector_mode
        self.min_samples = min_samples
        self.traces: deque[TraceEvent] = deque(maxlen=max_traces)
        self.latency: dict[LatencyMetric, LatencyStats] = {
            m: LatencyStats(min_samples) for m in LatencyMetric
        }
        self.frames_seen = 0
        self.frames_traced = 0

    def point(
        self,
        stage: str,
        *,
        t_ms: float | None = None,
        duration_ms: float | None = None,
        provider: str | None = None,
        status: TraceStatus = TraceStatus.OK,
        generation_id: int | None = None,
        detail: dict[str, Any] | None = None,
        emit: bool = True,
    ) -> TraceEvent:
        t = self.bus.clock.now_ms() if t_ms is None else t_ms
        ev = TraceEvent(
            session_id=self.bus.session_id,
            trial_id=self.trial_id,
            trace_id=new_id("trc"),
            stage=stage,
            timestamp_monotonic_ms=round(t, 3),
            duration_ms=None if duration_ms is None else round(duration_ms, 3),
            detector_mode=self.detector_mode,
            provider=provider,
            status=status,
            generation_id=generation_id,
            detail=detail,
        )
        self.traces.append(ev)
        if emit:
            self.bus.emit(EnvelopeKind.TRACE, ev, generation_id=generation_id, t_ms=t)
        return ev

    def sample(
        self,
        metric: LatencyMetric,
        value_ms: float | None,
        status: TraceStatus = TraceStatus.OK,
        generation_id: int | None = None,
        emit: bool = True,
    ) -> None:
        self.latency[metric].add(value_ms, status)
        if emit:
            self.bus.emit(
                EnvelopeKind.LATENCY,
                LatencyPayload(
                    metric=metric,
                    value_ms=None if value_ms is None else round(value_ms, 3),
                    status=status,
                    generation_id=generation_id,
                ),
                generation_id=generation_id,
            )

    def frame_processed(self, inference_ms: float | None) -> None:
        """Per-frame detector timing: recorded for percentiles, not emitted (50 Hz)."""
        self.frames_seen += 1
        if inference_ms is not None:
            self.frames_traced += 1
            self.latency[LatencyMetric.DETECTOR_INFERENCE].add(inference_ms, TraceStatus.OK)

    def summary(self) -> LatencySummaryPayload:
        return LatencySummaryPayload(metrics={m: s.summary() for m, s in self.latency.items()})

    def emit_summary(self) -> None:
        self.bus.emit(EnvelopeKind.LATENCY_SUMMARY, self.summary())

    @property
    def frame_trace_coverage(self) -> float | None:
        return self.frames_traced / self.frames_seen if self.frames_seen else None
