"""Trial records: what happened, under which configuration, and how it scored.

A trial record is the unit of evidence. It stores the server-accepted configuration, the
event stream (no raw audio, no audio levels; transcripts only when export was enabled), raw
latency samples so aggregates can be recomputed later, annotations when the trial is
labelled, and the computed summary.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from turnsense.events import (
    DecisionType,
    EffectiveConfig,
    Envelope,
    EnvelopeKind,
    LatencyMetric,
    Transport,
    TrialCounts,
    TrialMetrics,
    TrialSummaryPayload,
    TrialType,
)
from turnsense.metrics.turn import (
    DEFAULT_TOLERANCE,
    ReferenceTurn,
    Tolerance,
    score_barge_ins,
    score_turn_ends,
)
from turnsense.tracing import iso, utc_now


class UserSegment(BaseModel):
    start_ms: float
    end_ms: float
    kind: Literal["speech", "backchannel"]
    turn_index: int | None = None


class ScriptAnnotations(BaseModel):
    """Ground truth for a labelled trial, on the session's audio timeline."""

    source: str  # "scenario_script" or "manifest:<id>"
    turns: list[tuple[float, float]]  # (start_ms, end_ms) of each user conversational turn
    segments: list[UserSegment] = Field(default_factory=list)


class ResolvedAnnotations(BaseModel):
    source: str
    turn_ends_ms: list[float]
    barge_ins_ms: list[float]  # user speech onsets that fell inside an agent output interval
    backchannels_ms: list[float]
    not_applicable: list[str] = Field(default_factory=list)


class TrialRecord(BaseModel):
    trial_id: str
    session_id: str
    evaluation_id: str | None = None
    created_at: str
    ended_at: str | None = None
    trial_type: TrialType
    transport: Transport
    scenario_id: str
    trial_label: str | None = None
    detector_mode: str
    model_version: str
    decision_policy_version: str
    status: Literal["running", "completed", "aborted", "error"]
    labelled: bool
    metrics: TrialMetrics = Field(default_factory=TrialMetrics)
    counts: TrialCounts = Field(default_factory=TrialCounts)
    effective_config: EffectiveConfig
    annotations: ResolvedAnnotations | None = None
    tolerance_ms: dict[str, float] | None = None
    latency_samples: dict[str, list[float]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    events: list[Envelope] = Field(default_factory=list)

    LIST_FIELDS: ClassVar[tuple[str, ...]] = (
        "trial_id", "session_id", "evaluation_id", "created_at", "ended_at", "trial_type",
        "transport", "scenario_id", "trial_label", "detector_mode", "model_version",
        "decision_policy_version", "status", "labelled", "metrics", "counts",
    )  # fmt: skip

    def list_item(self) -> dict[str, Any]:
        return self.model_dump(mode="json", include=set(self.LIST_FIELDS))

    def detail(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"latency_samples"})


class TrialRecorder:
    """Bus listener that keeps the exportable event stream for one trial."""

    def __init__(self, keep_transcripts: bool, limit: int = 50_000) -> None:
        self.keep_transcripts = keep_transcripts
        self.limit = limit
        self.events: list[Envelope] = []
        self.truncated = False

    def __call__(self, env: Envelope) -> None:
        if env.kind == EnvelopeKind.AUDIO_LEVEL:
            return
        if env.kind == EnvelopeKind.TRANSCRIPT and not self.keep_transcripts:
            return
        if len(self.events) >= self.limit:
            self.truncated = True
            return
        self.events.append(env)


def _p(values: list[float], q: float, min_samples: int) -> float | None:
    from turnsense.metrics.stats import percentile

    need = 1 if q == 50 else min_samples
    return None if len(values) < need else percentile(values, q)


def resolve_annotations(
    script: ScriptAnnotations, generations: list[tuple[float, float | None]]
) -> ResolvedAnnotations:
    """Derive interruption references from the script and the agent's actual output intervals.

    A user speech onset inside an agent output interval (committed TURN_COMPLETE → response
    finished/cancelled) is a genuine interruption the system should act on; a backchannel inside
    one is a reference *non*-interruption.
    """
    barge, back, na = [], [], []
    for seg in script.segments:
        inside = any(
            start <= seg.start_ms < (end if end is not None else float("inf"))
            for start, end in generations
        )
        if seg.kind == "backchannel":
            if inside:
                back.append(seg.start_ms)
            else:
                na.append(f"backchannel at {seg.start_ms:.0f} ms was not during agent output")
        elif inside:
            barge.append(seg.start_ms)
    return ResolvedAnnotations(
        source=script.source,
        turn_ends_ms=[end for _, end in script.turns],
        barge_ins_ms=barge,
        backchannels_ms=back,
        not_applicable=na,
    )


def build_summary(
    *,
    trial_id: str,
    session_id: str,
    trial_type: TrialType,
    config: EffectiveConfig,
    orchestrator: Any,  # turnsense.orchestrator.session.Orchestrator (avoids an import cycle)
    script: ScriptAnnotations | None,
    status: Literal["completed", "aborted", "error"],
    dropped_frames: int = 0,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
    min_samples: int = 20,
) -> tuple[TrialSummaryPayload, ResolvedAnnotations | None, dict[str, list[float]]]:
    orch = orchestrator
    tracer = orch.tracer
    samples = {m.value: [round(v, 3) for v in tracer.latency[m].ok] for m in LatencyMetric}
    notes: list[str] = []

    turn_commits = [
        d.frame_end_ms for d in orch.decisions if d.decision == DecisionType.TURN_COMPLETE
    ]
    barge_commits = [d for d in orch.decisions if d.decision == DecisionType.BARGE_IN]
    auto_barge = [d.frame_end_ms for d in barge_commits if not d.manual]
    generations = [(g.decided_ms, g.ended_ms) for g in orch.generations]

    m = TrialMetrics(
        stale_chunks_played=orch.stale_chunks_played,
        stale_chunks_rejected=orch.stale_chunks_rejected,
        dropped_frames=dropped_frames,
        frame_trace_coverage=tracer.frame_trace_coverage,
    )
    resp = samples[LatencyMetric.RESPONSE_START.value]
    det = samples[LatencyMetric.DETECTOR_INFERENCE.value]
    cancel = samples[LatencyMetric.BARGE_IN_CANCEL.value]
    m.response_start_p50_ms = _p(resp, 50, min_samples)
    m.response_start_p95_ms = _p(resp, 95, min_samples)
    m.response_start_p99_ms = _p(resp, 99, min_samples)
    m.detector_inference_p50_ms = _p(det, 50, min_samples)
    m.detector_inference_p95_ms = _p(det, 95, min_samples)
    m.detector_inference_p99_ms = _p(det, 99, min_samples)
    m.barge_in_cancel_p50_ms = _p(cancel, 50, min_samples)

    counts = TrialCounts(
        frames=orch.counters.frames,
        committed_turn_ends=len(turn_commits),
        committed_barge_ins=len(barge_commits),
        agent_speaking_intervals=len(orch.agent_intervals),
        responses_started=orch.counters.responses_started,
        responses_cancelled=orch.counters.responses_cancelled,
        errors=orch.counters.errors,
    )

    resolved = None
    if script is not None:
        resolved = resolve_annotations(script, generations)
        refs = [ReferenceTurn(s, e) for s, e in script.turns]
        ts = score_turn_ends(refs, turn_commits, tolerance)
        bs = score_barge_ins(generations, auto_barge, resolved.barge_ins_ms, tolerance)
        m.turn_precision, m.turn_recall, m.turn_f1 = ts.precision, ts.recall, ts.f1
        m.premature_end_rate, m.missed_end_rate = ts.premature_end_rate, ts.missed_end_rate
        samples[LatencyMetric.TURN_END_DELAY.value] = [round(d, 3) for d in ts.delays_ms]
        m.turn_end_delay_p50_ms = _p(ts.delays_ms, 50, min_samples)
        m.turn_end_delay_p95_ms = _p(ts.delays_ms, 95, min_samples)
        # rates over agent output intervals (= responses started)
        m.false_barge_in_rate = bs.false_barge_ins / len(generations) if generations else None
        m.missed_barge_in_rate = bs.missed_barge_in_rate
        counts.reference_turn_ends = ts.n_reference
        counts.reference_barge_ins = bs.reference_barge_ins
        notes.extend(resolved.not_applicable)
        if any(d.manual for d in barge_commits):
            notes.append("Manual interrupts are excluded from barge-in scoring.")
    else:
        notes.append("Unlabelled trial: task-quality metrics are not available.")
    if config.runtime_profile == "simulation":
        notes.append(
            "Simulated providers: STT/LLM/TTS latencies are modelled, not measured; "
            "audio is synthetic and says nothing about real speakers."
        )
    if m.stale_chunks_played:
        notes.append(f"{m.stale_chunks_played} stale chunk(s) were played after cancellation.")

    summary = TrialSummaryPayload(
        trial_id=trial_id,
        session_id=session_id,
        trial_type=trial_type,
        scenario_id=config.scenario_id,
        detector_mode=config.detector_mode,
        model_version=config.model_version,
        decision_policy_version=config.decision_policy_version,
        labelled=script is not None,
        status=status,
        metrics=m,
        counts=counts,
        tolerance_ms=tolerance.as_dict() if script is not None else None,
        notes=notes,
    )
    return summary, resolved, samples


class TrialStore:
    """One JSON file per trial under ``data/trials``; an in-memory index for listing.

    Several processes may write (API + LiveKit worker); the index rescans when the directory
    changes.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] = {}
        self._seen_mtime_ns = -1
        self._refresh()

    def _refresh(self) -> None:
        try:
            mtime = self.root.stat().st_mtime_ns
        except FileNotFoundError:
            return
        if mtime == self._seen_mtime_ns:
            return
        index: dict[str, dict[str, Any]] = {}
        for path in self.root.glob("tri_*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                index[raw["trial_id"]] = {k: raw.get(k) for k in TrialRecord.LIST_FIELDS}
            except (OSError, ValueError, KeyError):
                continue
        self._index = index
        self._seen_mtime_ns = mtime

    def save(self, record: TrialRecord) -> None:
        path = self.root / f"{record.trial_id}.json"
        tmp = path.with_suffix(".json.tmp")
        data = record.model_dump_json()
        with self._lock:
            stale_index = self._seen_mtime_ns != self.root.stat().st_mtime_ns
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)
            self._index[record.trial_id] = record.list_item()
            if not stale_index:  # our own write: no need to rescan every file
                self._seen_mtime_ns = self.root.stat().st_mtime_ns

    def get(self, trial_id: str) -> TrialRecord | None:
        if not trial_id.startswith("tri_") or not trial_id[4:].isalnum():
            return None
        path = self.root / f"{trial_id}.json"
        if not path.is_file():
            return None
        return TrialRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(
        self,
        *,
        detector_mode: str | None = None,
        scenario_id: str | None = None,
        trial_type: str | None = None,
        evaluation_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[int, list[dict[str, Any]]]:
        with self._lock:
            self._refresh()
            items = list(self._index.values())
        if detector_mode:
            items = [i for i in items if i["detector_mode"] == detector_mode]
        if scenario_id:
            items = [i for i in items if i["scenario_id"] == scenario_id]
        if trial_type:
            items = [i for i in items if i["trial_type"] == trial_type]
        if evaluation_id:
            items = [i for i in items if i["evaluation_id"] == evaluation_id]
        items.sort(key=lambda i: i["created_at"] or "", reverse=True)
        return len(items), items[offset : offset + limit]


def now_iso() -> str:
    return iso(utc_now())
