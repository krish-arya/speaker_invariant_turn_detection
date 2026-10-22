"""Wire contract shared by the worker, the API server, exports and the frontend.

Everything that crosses a process or network boundary is defined here so that the
baseline and learned detector variants are indistinguishable to their consumers
(architecture.md §2.6, project.md FR-07). See docs/api-contract.md for prose.

Conventions:
* JSON field names are snake_case on the wire; the frontend normalises at its boundary.
* ``*_ms`` timestamps are milliseconds on the session's monotonic clock, measured from
  the session origin. Wall-clock time appears only in ``wall_time``/``*_at`` fields and is
  never mixed into durations (architecture.md §4.11).
* Values a component cannot measure are ``None`` — never a fabricated 0 or confidence.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DetectorMode(StrEnum):
    SILENCE_BASELINE = "silence_baseline"
    LEARNED_CONTROL = "learned_control"
    ANTI_TRANSFER = "anti_transfer"


class TurnEventType(StrEnum):
    """Detector predictions (architecture.md §5.3)."""

    CONTINUE = "CONTINUE"
    POSSIBLE_END = "POSSIBLE_END"
    TURN_COMPLETE = "TURN_COMPLETE"
    BARGE_IN = "BARGE_IN"
    BACKCHANNEL = "BACKCHANNEL"


class DecisionType(StrEnum):
    """Decisions committed by the orchestrator's policy — distinct from predictions."""

    TURN_COMPLETE = "TURN_COMPLETE"
    BARGE_IN = "BARGE_IN"
    BACKCHANNEL_IGNORED = "BACKCHANNEL_IGNORED"


class SessionState(StrEnum):
    """Server-side session states (architecture.md §6, frontend.md §7).

    ``idle`` and ``requesting_permission`` exist only in the browser.
    """

    CONNECTING = "connecting"
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    POSSIBLE_END = "possible_end"
    GENERATING = "generating"
    AGENT_SPEAKING = "agent_speaking"
    INTERRUPTED = "interrupted"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    ENDING = "ending"
    ENDED = "ended"


class Transport(StrEnum):
    SIMULATION = "simulation"
    LIVEKIT = "livekit"


class TrialType(StrEnum):
    SIMULATED = "simulated"  # synthetic scripted audio driven through the real pipeline
    REPLAY = "replay"  # labelled recorded clips replayed through the real pipeline
    LIVE = "live"  # a human speaking through the browser — unlabelled


class TraceStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ErrorSource(StrEnum):
    TRANSPORT = "transport"
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    DETECTOR = "detector"
    VAD = "vad"
    WORKER = "worker"


class LatencyMetric(StrEnum):
    DETECTOR_INFERENCE = "detector_inference"
    TURN_END_DELAY = "turn_end_delay"  # annotated end -> committed decision (labelled only)
    RESPONSE_START = "response_start"  # last user frame used -> first response audio delivered
    BARGE_IN_CANCEL = "barge_in_cancel"  # barge-in committed -> playback flush acknowledged
    STT_FINAL = "stt_final"
    LLM_FIRST_TOKEN = "llm_first_token"
    TTS_FIRST_AUDIO = "tts_first_audio"


class EnvelopeKind(StrEnum):
    HELLO = "hello"
    SESSION_STATUS = "session_status"
    VAD = "vad"
    TURN_PREDICTION = "turn_prediction"
    TURN_DECISION = "turn_decision"
    TRANSCRIPT = "transcript"
    TRACE = "trace"
    LATENCY = "latency"
    LATENCY_SUMMARY = "latency_summary"
    AUDIO_LEVEL = "audio_level"
    PLAYBACK = "playback"
    TRIAL_SUMMARY = "trial_summary"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Configuration that is frozen per trial
# ---------------------------------------------------------------------------


class PolicyConfig(_Model):
    """Decision-policy parameters. Recorded verbatim in every trial (goal.md §6)."""

    version: str = "policy-0.1"
    # silence_baseline: silence after speech before committing TURN_COMPLETE.
    silence_timeout_ms: float = Field(700.0, ge=50, le=5000)
    # silence after speech before predicting POSSIBLE_END (both baseline and learned).
    possible_end_ms: float = Field(150.0, ge=0, le=5000)
    # learned detectors: probability thresholds on p(turn end).
    complete_threshold: float = Field(0.6, ge=0.0, le=1.0)
    possible_end_threshold: float = Field(0.3, ge=0.0, le=1.0)
    # consecutive TURN_COMPLETE predictions required before the orchestrator commits.
    consecutive_frames: int = Field(3, ge=1, le=100)
    # minimum silence before any TURN_COMPLETE may be committed (endpointing guard).
    min_silence_guard_ms: float = Field(100.0, ge=0, le=5000)
    # learned detectors: commit anyway after this much silence (bounded worst case).
    max_silence_ms: float = Field(2000.0, ge=100, le=10000)
    # user speech while the agent speaks must last this long to count as barge-in.
    barge_in_min_speech_ms: float = Field(200.0, ge=0, le=3000)
    # speech this short that ends during agent output is treated as a backchannel.
    backchannel_max_ms: float = Field(600.0, ge=0, le=3000)
    # ignore speech segments shorter than this when listening (clicks, coughs).
    min_speech_ms: float = Field(60.0, ge=0, le=1000)


class VadConfig(_Model):
    kind: Literal["silero", "energy"] = "energy"
    threshold: float = 0.5
    # hangover smooths VAD flicker; it does not decide turn completion.
    min_speech_frames: int = 2
    min_silence_frames: int = 2


class ProviderInfo(_Model):
    """Non-secret provider identifiers; never contains keys or URLs with secrets."""

    stt: str
    llm: str
    tts: str


class EffectiveConfig(_Model):
    """The configuration the server actually accepted for a trial (frontend.md §6.2)."""

    detector_mode: DetectorMode
    model_version: str
    decision_policy_version: str
    policy: PolicyConfig
    vad: VadConfig
    frame_ms: int = 20
    sample_rate: int = 16000
    providers: ProviderInfo
    runtime_profile: Literal["simulation", "hosted", "local"]
    transport: Transport
    scenario_id: str
    seed: int | None = None
    transcript_enabled: bool
    transcript_export_enabled: bool
    raw_audio_retention: bool = False
    # start the LLM during POSSIBLE_END on a stable partial transcript (reused only if unchanged)
    speculative_prefill: bool = False
    server_version: str
    config_hash: str


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


class SessionInfo(_Model):
    session_id: str
    trial_id: str
    transport: Transport
    trial_type: TrialType
    scenario_id: str
    trial_label: str | None = None
    effective_config: EffectiveConfig
    started_at: str


class HelloPayload(_Model):
    protocol_version: int = PROTOCOL_VERSION
    server_version: str
    session: SessionInfo


class SessionStatusPayload(_Model):
    state: SessionState
    previous_state: SessionState | None
    reason: str | None = None


class VadPayload(_Model):
    """Speech-presence transitions only — never a turn decision."""

    speaking: bool
    probability: float | None
    frame_end_ms: float


class TurnEvent(_Model):
    """A detector prediction (architecture.md §7.1)."""

    session_id: str
    sequence: int
    event_type: TurnEventType
    detector_mode: DetectorMode
    model_version: str
    event_end_ms: float
    inference_ms: float | None
    confidence: float | None
    decision_policy_version: str
    reason_codes: list[str] = Field(default_factory=list)


class TurnDecisionPayload(_Model):
    """A committed orchestrator decision."""

    decision: DecisionType
    source_event_sequence: int | None
    # last user audio frame the decision was based on (start of the response-latency clock)
    last_user_frame_ms: float
    committed_ms: float
    confidence: float | None
    reason_codes: list[str] = Field(default_factory=list)


class TranscriptPayload(_Model):
    segment_id: str
    role: Literal["user", "assistant"]
    text: str
    is_final: bool
    revision: int
    start_ms: float | None = None
    end_ms: float | None = None


class TraceEvent(_Model):
    """A trace span or point (architecture.md §7.2)."""

    session_id: str
    trial_id: str | None
    trace_id: str
    stage: str
    timestamp_monotonic_ms: float
    duration_ms: float | None = None
    detector_mode: DetectorMode | None = None
    provider: str | None = None
    status: TraceStatus = TraceStatus.OK
    generation_id: int | None = None
    detail: dict[str, Any] | None = None


class LatencyPayload(_Model):
    metric: LatencyMetric
    value_ms: float | None
    status: TraceStatus = TraceStatus.OK
    generation_id: int | None = None


class PercentileSummary(_Model):
    count: int  # valid (status ok) samples
    failed: int = 0  # error/timeout/cancelled samples, excluded from percentiles
    latest_ms: float | None = None
    p50_ms: float | None = None  # None until count >= min_samples
    p95_ms: float | None = None
    p99_ms: float | None = None
    min_samples: int


class LatencySummaryPayload(_Model):
    metrics: dict[LatencyMetric, PercentileSummary]


class AudioLevelPayload(_Model):
    source: Literal["user", "agent"]
    rms_dbfs: float


class PlaybackPayload(_Model):
    action: Literal["start", "end", "flush"]
    reason: str | None = None


class ErrorPayload(_Model):
    source: ErrorSource
    code: str
    message: str
    recoverable: bool


class TrialMetrics(_Model):
    """Per-trial metrics. ``None`` means not measurable for this trial (e.g. no labels)."""

    turn_precision: float | None = None
    turn_recall: float | None = None
    turn_f1: float | None = None
    premature_end_rate: float | None = None
    missed_end_rate: float | None = None
    turn_end_delay_p50_ms: float | None = None
    turn_end_delay_p95_ms: float | None = None
    response_start_p50_ms: float | None = None
    response_start_p95_ms: float | None = None
    response_start_p99_ms: float | None = None
    detector_inference_p50_ms: float | None = None
    detector_inference_p95_ms: float | None = None
    detector_inference_p99_ms: float | None = None
    false_barge_in_rate: float | None = None
    missed_barge_in_rate: float | None = None
    barge_in_cancel_p50_ms: float | None = None
    stale_chunks_played: int = 0
    stale_chunks_rejected: int = 0
    dropped_frames: int = 0
    frame_trace_coverage: float | None = None


class TrialCounts(_Model):
    frames: int = 0
    reference_turn_ends: int = 0
    committed_turn_ends: int = 0
    reference_barge_ins: int = 0
    committed_barge_ins: int = 0
    agent_speaking_intervals: int = 0
    responses_started: int = 0
    responses_cancelled: int = 0
    errors: int = 0


class TrialSummaryPayload(_Model):
    trial_id: str
    session_id: str
    trial_type: TrialType
    scenario_id: str
    detector_mode: DetectorMode
    model_version: str
    decision_policy_version: str
    labelled: bool
    status: Literal["completed", "aborted", "error"]
    metrics: TrialMetrics
    counts: TrialCounts
    tolerance_ms: dict[str, float] | None = None
    notes: list[str] = Field(default_factory=list)


PAYLOAD_TYPES: dict[EnvelopeKind, type[_Model]] = {
    EnvelopeKind.HELLO: HelloPayload,
    EnvelopeKind.SESSION_STATUS: SessionStatusPayload,
    EnvelopeKind.VAD: VadPayload,
    EnvelopeKind.TURN_PREDICTION: TurnEvent,
    EnvelopeKind.TURN_DECISION: TurnDecisionPayload,
    EnvelopeKind.TRANSCRIPT: TranscriptPayload,
    EnvelopeKind.TRACE: TraceEvent,
    EnvelopeKind.LATENCY: LatencyPayload,
    EnvelopeKind.LATENCY_SUMMARY: LatencySummaryPayload,
    EnvelopeKind.AUDIO_LEVEL: AudioLevelPayload,
    EnvelopeKind.PLAYBACK: PlaybackPayload,
    EnvelopeKind.TRIAL_SUMMARY: TrialSummaryPayload,
    EnvelopeKind.ERROR: ErrorPayload,
}


class Envelope(_Model):
    """One message on the frontend update channel (WebSocket or LiveKit data channel)."""

    v: int = PROTOCOL_VERSION
    kind: EnvelopeKind
    session_id: str
    seq: int  # per-session, strictly increasing from 1; gaps mean lost messages
    t_ms: float  # session-monotonic time the message was produced
    wall_time: str  # ISO-8601 UTC, for display/export only
    generation_id: int | None = None
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# REST bodies
# ---------------------------------------------------------------------------


class CreateSessionRequest(_Model):
    transport: Transport = Transport.SIMULATION
    detector_mode: DetectorMode = DetectorMode.SILENCE_BASELINE
    scenario_id: str = "pause_then_continue"
    policy_overrides: dict[str, Any] | None = None
    transcript_enabled: bool = True
    transcript_export_enabled: bool = False
    speculative_prefill: bool = False
    trial_label: str | None = Field(None, max_length=120)
    seed: int | None = None
    client_build: str | None = Field(None, max_length=64)


class LiveKitJoinInfo(_Model):
    url: str
    room: str
    token: str  # short-lived room grant — never logged or displayed
    expires_at: str


class CreateSessionResponse(_Model):
    session_id: str
    trial_id: str
    effective_config: EffectiveConfig
    events_url: str | None  # WebSocket path for the simulation transport
    livekit: LiveKitJoinInfo | None = None


class CreateEvaluationRequest(_Model):
    detector_modes: list[DetectorMode] = Field(min_length=1, max_length=3)
    scenario_ids: list[str] = Field(min_length=1, max_length=10)
    repeats: int = Field(3, ge=1, le=50)
    seed: int = 0
    policy_overrides: dict[str, Any] | None = None
    speculative_prefill: bool = False
    trial_label: str | None = Field(None, max_length=120)


class EvaluationStatus(_Model):
    evaluation_id: str
    status: Literal["queued", "running", "completed", "stopped", "error"]
    total: int
    completed: int
    trial_ids: list[str]
    error: str | None = None
    created_at: str
