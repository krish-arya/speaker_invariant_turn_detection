/**
 * Live session view state (frontend.md §7). The UI never guesses state: phases come
 * from `session_status` envelopes and transport events; predictions and committed
 * decisions are kept apart; stale/duplicate/foreign messages are dropped and counted.
 */
import type {
  CreateSessionResponse,
  EffectiveConfig,
  Envelope,
  ErrorPayload,
  LatencyMetric,
  ParseResult,
  PercentileSummary,
  ServerState,
  SessionInfo,
  TraceStatus,
  Transport,
  TrialSummary,
  TurnDecision,
  TurnEvent,
} from "../api/schemas";
import { formatMs, formatRelative } from "../lib/format";
import {
  DECISION_LABELS,
  DETECTOR_LABELS,
  ERROR_SOURCE_LABELS,
  ERROR_SOURCE_TO_TIMELINE,
  PHASE_META,
  STATE_LABELS,
  TRACE_SOURCE,
  TRACE_TIMELINE_LABELS,
  TURN_EVENT_LABELS,
  type TimelineSource,
  type Tone,
  type UiPhase,
} from "../lib/labels";

export const TIMELINE_LIMIT = 500;
export const TRANSCRIPT_LIMIT = 200;
export const SEEN_SEQ_LIMIT = 2048;
export const ERROR_LIMIT = 20;
export const CANCELLED_GENERATION_LIMIT = 64;

export type TransportStatus = "idle" | "connecting" | "connected" | "reconnecting" | "disconnected" | "error";

export type TimelineCategory = "turn" | "pipeline" | "error" | "session";

export interface TimelineItem {
  id: string;
  type: "event" | "gap" | "local";
  category: TimelineCategory;
  source: TimelineSource;
  /** Predictions and committed decisions render differently so they are never confused. */
  emphasis: "prediction" | "decision" | "state" | "info";
  label: string;
  code: string | null;
  detail: string | null;
  confidence: number | null;
  reasonCodes: string[];
  tMs: number | null;
  wallTime: string | null;
  seq: number | null;
  generationId: number | null;
  tone: Tone;
  late: boolean;
  meta: Record<string, string> | null;
  raw: unknown;
}

export interface TranscriptSegment {
  segmentId: string;
  role: "user" | "assistant";
  text: string;
  isFinal: boolean;
  revision: number;
  startMs: number | null;
  endMs: number | null;
  generationId: number | null;
  /** Assistant output of a cancelled generation: shown as interrupted, never as delivered. */
  interrupted: boolean;
}

export interface LatestSample {
  valueMs: number | null;
  status: TraceStatus;
  tMs: number;
  generationId: number | null;
}

export interface StageSample {
  durationMs: number | null;
  status: TraceStatus;
  tMs: number;
  provider: string | null;
}

export interface Diagnostics {
  received: number;
  accepted: number;
  unknownKinds: Record<string, number>;
  unknownTotal: number;
  invalid: number;
  unsupportedVersion: number;
  duplicates: number;
  foreign: number;
  outOfOrder: number;
  gaps: number;
  missingMessages: number;
  staleDropped: number;
  lastProblem: string | null;
}

export interface LocalError {
  kind: "permission_denied" | "no_device" | "join_failed" | "api" | "transport" | "protocol" | "interrupt";
  message: string;
  detail?: string;
  retryable: boolean;
  /** fatal errors move the session to the `error` phase; others are notices */
  fatal: boolean;
}

export interface SessionViewState {
  phase: UiPhase;
  serverState: ServerState | null;
  transport: Transport | null;
  transportStatus: TransportStatus;
  transportDetail: string | null;
  isMock: boolean;
  sessionId: string | null;
  trialId: string | null;
  session: SessionInfo | null;
  effectiveConfig: EffectiveConfig | null;
  serverProtocolVersion: number | null;
  highestSeq: number;
  seenSeqs: number[];
  timeline: TimelineItem[];
  timelineTrimmed: number;
  localCounter: number;
  vad: { speaking: boolean; probability: number | null; tMs: number } | null;
  prediction: TurnEvent | null;
  decision: (TurnDecision & { generationId: number | null; tMs: number }) | null;
  transcript: TranscriptSegment[];
  latencyLatest: Partial<Record<LatencyMetric, LatestSample>>;
  latencySummary: Partial<Record<LatencyMetric, PercentileSummary>>;
  stageLatest: Record<string, StageSample>;
  audioLevel: { user: number | null; agent: number | null };
  currentGeneration: number | null;
  cancelledGenerations: number[];
  /** "Stopping response…" until the cancellation is acknowledged */
  stopping: boolean;
  playing: boolean;
  trialSummary: TrialSummary | null;
  errors: Array<ErrorPayload & { tMs: number; seq: number }>;
  localError: LocalError | null;
  diagnostics: Diagnostics;
  lastTMs: number | null;
}

export type SessionAction =
  | { type: "start_requested"; transport: Transport; isMock: boolean; needsPermission: boolean }
  | { type: "permission_granted" }
  | { type: "session_created"; response: CreateSessionResponse }
  | { type: "transport_status"; status: TransportStatus; detail?: string | null }
  | { type: "envelopes"; results: ParseResult[] }
  | { type: "local_error"; error: LocalError }
  | { type: "end_requested" }
  | { type: "interrupt_requested" }
  | { type: "interrupt_failed"; message: string }
  | { type: "source_closed" }
  | { type: "clear_view" }
  | { type: "dismiss_error" };

export const emptyDiagnostics = (): Diagnostics => ({
  received: 0,
  accepted: 0,
  unknownKinds: {},
  unknownTotal: 0,
  invalid: 0,
  unsupportedVersion: 0,
  duplicates: 0,
  foreign: 0,
  outOfOrder: 0,
  gaps: 0,
  missingMessages: 0,
  staleDropped: 0,
  lastProblem: null,
});

export function initialSessionState(): SessionViewState {
  return {
    phase: "idle",
    serverState: null,
    transport: null,
    transportStatus: "idle",
    transportDetail: null,
    isMock: false,
    sessionId: null,
    trialId: null,
    session: null,
    effectiveConfig: null,
    serverProtocolVersion: null,
    highestSeq: 0,
    seenSeqs: [],
    timeline: [],
    timelineTrimmed: 0,
    localCounter: 0,
    vad: null,
    prediction: null,
    decision: null,
    transcript: [],
    latencyLatest: {},
    latencySummary: {},
    stageLatest: {},
    audioLevel: { user: null, agent: null },
    currentGeneration: null,
    cancelledGenerations: [],
    stopping: false,
    playing: false,
    trialSummary: null,
    errors: [],
    localError: null,
    diagnostics: emptyDiagnostics(),
    lastTMs: null,
  };
}

// ---------------------------------------------------------------------------
// Selectors
// ---------------------------------------------------------------------------

const RESPONSE_PHASES: UiPhase[] = ["generating", "agent_speaking"];

/** Transport reconnection overrides the displayed phase without inventing server state. */
export function selectDisplayPhase(s: SessionViewState): UiPhase {
  if (s.transportStatus === "reconnecting" && !["idle", "ended", "error", "ending"].includes(s.phase)) {
    return "reconnecting";
  }
  return s.phase;
}

/** True while configuration must stay frozen (frontend.md §6.2). */
export function isSessionBusy(s: SessionViewState): boolean {
  if (["requesting_permission", "connecting", "ending"].includes(s.phase)) return true;
  if (s.phase === "idle" || s.phase === "ended") return false;
  return s.sessionId !== null && ["connecting", "connected", "reconnecting"].includes(s.transportStatus);
}

export function canInterrupt(s: SessionViewState): boolean {
  return RESPONSE_PHASES.includes(s.phase) && !s.stopping;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function baseItem(partial: Partial<TimelineItem> & Pick<TimelineItem, "id" | "label" | "source" | "category">): TimelineItem {
  return {
    type: "event",
    emphasis: "info",
    code: null,
    detail: null,
    confidence: null,
    reasonCodes: [],
    tMs: null,
    wallTime: null,
    seq: null,
    generationId: null,
    tone: "neutral",
    late: false,
    meta: null,
    raw: null,
    ...partial,
  };
}

function pushItems(state: SessionViewState, items: TimelineItem[]): SessionViewState {
  if (items.length === 0) return state;
  let timeline = state.timeline.concat(items);
  let trimmed = state.timelineTrimmed;
  if (timeline.length > TIMELINE_LIMIT) {
    trimmed += timeline.length - TIMELINE_LIMIT;
    timeline = timeline.slice(timeline.length - TIMELINE_LIMIT);
  }
  return { ...state, timeline, timelineTrimmed: trimmed };
}

function localItem(
  state: SessionViewState,
  partial: Omit<Partial<TimelineItem>, "id"> & Pick<TimelineItem, "label" | "source" | "category">,
): SessionViewState {
  const n = state.localCounter + 1;
  const item = baseItem({ ...partial, id: `local-${n}`, type: partial.type ?? "local", tMs: partial.tMs ?? state.lastTMs });
  return pushItems({ ...state, localCounter: n }, [item]);
}

const PREDICTION_TONE: Record<TurnEvent["eventType"], Tone> = {
  CONTINUE: "active",
  POSSIBLE_END: "uncertain",
  TURN_COMPLETE: "uncertain",
  BARGE_IN: "uncertain",
  BACKCHANNEL: "neutral",
};

const DECISION_TONE: Record<TurnDecision["decision"], Tone> = {
  TURN_COMPLETE: "complete",
  BARGE_IN: "danger",
  BACKCHANNEL_IGNORED: "neutral",
};

/** The human-readable timeline line for an envelope, or null if it does not get one. */
export function timelineItemFor(env: Envelope, raw: unknown): TimelineItem | null {
  const common = {
    id: `e${env.seq}`,
    tMs: env.tMs,
    wallTime: env.wallTime,
    seq: env.seq,
    generationId: env.generationId,
    raw,
  };
  switch (env.kind) {
    case "hello": {
      const s = env.payload.session;
      const c = s.effectiveConfig;
      return baseItem({
        ...common,
        category: "session",
        source: "Session",
        label: "Session started",
        detail: `${DETECTOR_LABELS[c.detectorMode]} · scenario ${s.scenarioId} · ${s.transport}`,
        meta: {
          "Detector mode": c.detectorMode,
          "Model version": c.modelVersion,
          "Policy version": c.decisionPolicyVersion,
          "Config hash": c.configHash,
        },
      });
    }
    case "session_status": {
      const st = env.payload.state;
      return baseItem({
        ...common,
        category: "turn",
        source: "Orchestrator",
        emphasis: "state",
        label: STATE_LABELS[st],
        detail: env.payload.reason,
        tone: PHASE_META[st].tone,
      });
    }
    case "vad":
      return baseItem({
        ...common,
        category: "turn",
        source: "VAD",
        label: env.payload.speaking ? "Speech detected" : "Silence detected",
        detail:
          "Speech presence only — not a turn decision" +
          (env.payload.probability !== null ? ` · p(speech) ${env.payload.probability.toFixed(2)}` : ""),
        tone: env.payload.speaking ? "active" : "neutral",
      });
    case "turn_prediction": {
      const p = env.payload;
      return baseItem({
        ...common,
        category: "turn",
        source: "Turn detector",
        emphasis: "prediction",
        label: `Predicted: ${TURN_EVENT_LABELS[p.eventType]}`,
        code: p.eventType,
        detail: p.inferenceMs !== null ? `Inference ${formatMs(p.inferenceMs)}` : null,
        confidence: p.confidence,
        reasonCodes: p.reasonCodes,
        tone: PREDICTION_TONE[p.eventType],
        meta: {
          "Detector mode": p.detectorMode,
          "Model version": p.modelVersion,
          "Policy version": p.decisionPolicyVersion,
          "Detector sequence": String(p.sequence),
          "Window end": formatRelative(p.eventEndMs),
        },
      });
    }
    case "turn_decision": {
      const d = env.payload;
      return baseItem({
        ...common,
        category: "turn",
        source: "Orchestrator",
        emphasis: "decision",
        label: `Committed: ${DECISION_LABELS[d.decision]}`,
        code: d.decision,
        detail: `Based on user audio up to ${formatRelative(d.lastUserFrameMs)}${
          d.sourceEventSequence !== null ? ` · from prediction #${d.sourceEventSequence}` : ""
        }`,
        confidence: d.confidence,
        reasonCodes: d.reasonCodes,
        tone: DECISION_TONE[d.decision],
      });
    }
    case "trace": {
      const t = env.payload;
      const label = TRACE_TIMELINE_LABELS[t.stage];
      if (!label) return null;
      const failed = t.status === "error" || t.status === "timeout";
      const parts = [
        t.durationMs !== null ? formatMs(t.durationMs) : null,
        t.provider,
        t.status !== "ok" ? t.status : null,
      ].filter(Boolean);
      return baseItem({
        ...common,
        category: failed ? "error" : "pipeline",
        source: TRACE_SOURCE[t.stage] ?? "Orchestrator",
        label,
        detail: parts.length ? parts.join(" · ") : null,
        tone: failed ? "danger" : t.stage === "cancel_acknowledged" ? "danger" : t.stage === "frame_gap" ? "uncertain" : "neutral",
        code: t.stage,
      });
    }
    case "playback": {
      const a = env.payload.action;
      return baseItem({
        ...common,
        category: "pipeline",
        source: "Playback",
        label: a === "start" ? "Agent audio started" : a === "end" ? "Agent audio finished" : "Playback flushed",
        detail: a === "flush" ? `Queued agent audio discarded${env.payload.reason ? ` · ${env.payload.reason}` : ""}` : env.payload.reason,
        tone: a === "flush" ? "danger" : a === "start" ? "agent" : "neutral",
      });
    }
    case "trial_summary":
      return baseItem({
        ...common,
        category: "session",
        source: "Session",
        label: "Trial summary ready",
        detail: `${env.payload.labelled ? "Labelled" : "Unlabelled"} ${env.payload.trialType} trial · status ${env.payload.status}`,
      });
    case "error": {
      const e = env.payload;
      return baseItem({
        ...common,
        category: "error",
        source: ERROR_SOURCE_TO_TIMELINE[e.source],
        label: `${ERROR_SOURCE_LABELS[e.source]} error`,
        detail: `${e.message} (${e.code})${e.recoverable ? " · recoverable" : " · not recoverable"}`,
        tone: "danger",
        code: e.code,
      });
    }
    default:
      return null;
  }
}

function upsertTranscript(list: TranscriptSegment[], seg: TranscriptSegment): TranscriptSegment[] {
  const idx = list.findIndex((s) => s.segmentId === seg.segmentId);
  if (idx >= 0) {
    const existing = list[idx]!;
    if (seg.revision < existing.revision) return list;
    const next = list.slice();
    next[idx] = seg;
    return next;
  }
  const next = list.concat(seg);
  return next.length > TRANSCRIPT_LIMIT ? next.slice(next.length - TRANSCRIPT_LIMIT) : next;
}

/** State effects of an in-order envelope (timeline item excluded). */
function applyEffects(state: SessionViewState, env: Envelope): SessionViewState {
  switch (env.kind) {
    case "hello": {
      const info = env.payload.session;
      return {
        ...state,
        session: info,
        effectiveConfig: info.effectiveConfig,
        sessionId: info.sessionId,
        trialId: info.trialId,
        transport: info.transport,
        serverProtocolVersion: env.payload.protocolVersion,
      };
    }
    case "session_status": {
      const st = env.payload.state;
      let phase: UiPhase = st;
      // Keep showing "Ending" after the user clicked End until the server confirms.
      if (state.phase === "ending" && st !== "ended" && st !== "error") phase = "ending";
      const keepStopping = st === "generating" || st === "agent_speaking" || st === "interrupted";
      return {
        ...state,
        phase,
        serverState: st,
        stopping: keepStopping ? state.stopping : false,
        playing: st === "agent_speaking" ? state.playing : st === "generating" ? state.playing : false,
        localError: phase === "error" ? state.localError : state.localError?.fatal ? null : state.localError,
      };
    }
    case "vad":
      return { ...state, vad: { speaking: env.payload.speaking, probability: env.payload.probability, tMs: env.tMs } };
    case "turn_prediction":
      return { ...state, prediction: env.payload };
    case "turn_decision": {
      const d = env.payload;
      const decision = { ...d, generationId: env.generationId, tMs: env.tMs };
      if (d.decision === "BARGE_IN") {
        const g = env.generationId ?? state.currentGeneration;
        const cancelled =
          g === null || state.cancelledGenerations.includes(g)
            ? state.cancelledGenerations
            : state.cancelledGenerations.concat(g).slice(-CANCELLED_GENERATION_LIMIT);
        const transcript =
          g === null
            ? state.transcript
            : state.transcript.map((seg) =>
                seg.role === "assistant" && seg.generationId === g && !seg.isFinal ? { ...seg, interrupted: true } : seg,
              );
        return { ...state, decision, cancelledGenerations: cancelled, stopping: true, transcript };
      }
      if (d.decision === "TURN_COMPLETE") {
        return { ...state, decision, currentGeneration: env.generationId ?? state.currentGeneration };
      }
      return { ...state, decision };
    }
    case "transcript": {
      const p = env.payload;
      return {
        ...state,
        transcript: upsertTranscript(state.transcript, {
          segmentId: p.segmentId,
          role: p.role,
          text: p.text,
          isFinal: p.isFinal,
          revision: p.revision,
          startMs: p.startMs,
          endMs: p.endMs,
          generationId: env.generationId,
          interrupted: false,
        }),
      };
    }
    case "trace": {
      const t = env.payload;
      let next = state;
      if (t.durationMs !== null || t.status !== "ok") {
        next = {
          ...next,
          stageLatest: {
            ...next.stageLatest,
            [t.stage]: { durationMs: t.durationMs, status: t.status, tMs: t.timestampMonotonicMs, provider: t.provider },
          },
        };
      }
      const gen = t.generationId ?? env.generationId;
      if (t.stage === "llm_request" && gen !== null && !next.cancelledGenerations.includes(gen)) {
        next = { ...next, currentGeneration: gen };
      }
      if (t.stage === "cancel_acknowledged") next = { ...next, stopping: false };
      return next;
    }
    case "latency": {
      const p = env.payload;
      return {
        ...state,
        latencyLatest: {
          ...state.latencyLatest,
          [p.metric]: { valueMs: p.valueMs, status: p.status, tMs: env.tMs, generationId: p.generationId ?? env.generationId },
        },
      };
    }
    case "latency_summary":
      return { ...state, latencySummary: { ...state.latencySummary, ...env.payload.metrics } };
    case "audio_level":
      return { ...state, audioLevel: { ...state.audioLevel, [env.payload.source]: env.payload.rmsDbfs } };
    case "playback": {
      const a = env.payload.action;
      if (a === "start") return { ...state, playing: true };
      if (a === "end") return { ...state, playing: false };
      return { ...state, playing: false, stopping: false };
    }
    case "trial_summary":
      return { ...state, trialSummary: env.payload };
    case "error": {
      const errors = state.errors.concat({ ...env.payload, tMs: env.tMs, seq: env.seq });
      return { ...state, errors: errors.length > ERROR_LIMIT ? errors.slice(errors.length - ERROR_LIMIT) : errors };
    }
    default:
      return state;
  }
}

function isStale(state: SessionViewState, env: Envelope): boolean {
  if (env.generationId === null || !state.cancelledGenerations.includes(env.generationId)) return false;
  if (env.kind === "transcript") return true;
  return env.kind === "playback" && env.payload.action === "start";
}

function applyResult(state: SessionViewState, result: ParseResult): SessionViewState {
  const d: Diagnostics = { ...state.diagnostics, received: state.diagnostics.received + 1 };
  if (!result.ok) {
    if (result.reason === "unknown_kind") {
      const k = result.kind ?? "?";
      d.unknownKinds = { ...d.unknownKinds, [k]: (d.unknownKinds[k] ?? 0) + 1 };
      d.unknownTotal += 1;
    } else if (result.reason === "unsupported_version") {
      d.unsupportedVersion += 1;
    } else {
      d.invalid += 1;
    }
    d.lastProblem = result.message;
    return { ...state, diagnostics: d };
  }

  const env = result.envelope;
  if (state.sessionId !== null && env.sessionId !== state.sessionId) {
    d.foreign += 1;
    d.lastProblem = `Dropped message for another session (${env.sessionId})`;
    return { ...state, diagnostics: d };
  }

  let late = false;
  if (env.seq <= state.highestSeq) {
    const oldest = state.seenSeqs.length ? state.seenSeqs[0]! : 0;
    if (state.seenSeqs.includes(env.seq) || env.seq < oldest) {
      d.duplicates += 1;
      return { ...state, diagnostics: d };
    }
    late = true;
    d.outOfOrder += 1;
  }

  const gapItems: TimelineItem[] = [];
  if (env.seq > state.highestSeq + 1) {
    const from = state.highestSeq + 1;
    const to = env.seq - 1;
    const missing = to - from + 1;
    d.gaps += 1;
    d.missingMessages += missing;
    gapItems.push(
      baseItem({
        id: `gap-${from}-${to}`,
        type: "gap",
        category: "error",
        source: "Transport",
        label: missing === 1 ? "1 message missing" : `${missing} messages missing`,
        detail: `Sequence ${from === to ? from : `${from}–${to}`} was not received. Events in this interval may be missing; nothing has been interpolated.`,
        tone: "uncertain",
        tMs: env.tMs,
      }),
    );
  }

  const seenSeqs = state.seenSeqs.concat(env.seq);
  let next: SessionViewState = {
    ...state,
    sessionId: state.sessionId ?? env.sessionId,
    highestSeq: Math.max(state.highestSeq, env.seq),
    seenSeqs: seenSeqs.length > SEEN_SEQ_LIMIT ? seenSeqs.slice(seenSeqs.length - SEEN_SEQ_LIMIT) : seenSeqs,
    lastTMs: late ? state.lastTMs : env.tMs,
    diagnostics: d,
  };

  if (isStale(next, env)) {
    next.diagnostics = { ...d, staleDropped: d.staleDropped + 1, lastProblem: `Ignored ${env.kind} for cancelled generation ${env.generationId}` };
    return pushItems(next, gapItems);
  }

  next.diagnostics = { ...d, accepted: d.accepted + 1 };
  const item = timelineItemFor(env, result.raw);
  if (late) {
    // A late message is shown (marked) but must not roll state back.
    return pushItems(next, item ? [...gapItems, { ...item, late: true }] : gapItems);
  }
  next = applyEffects(next, env);
  return pushItems(next, item ? [...gapItems, item] : gapItems);
}

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

export function sessionReducer(state: SessionViewState, action: SessionAction): SessionViewState {
  switch (action.type) {
    case "start_requested": {
      const fresh = initialSessionState();
      return {
        ...fresh,
        phase: action.needsPermission ? "requesting_permission" : "connecting",
        transport: action.transport,
        isMock: action.isMock,
        transportStatus: action.needsPermission ? "idle" : "connecting",
      };
    }
    case "permission_granted":
      return state.phase === "requesting_permission" ? { ...state, phase: "connecting", transportStatus: "connecting" } : state;
    case "session_created": {
      const r = action.response;
      return {
        ...state,
        sessionId: r.sessionId,
        trialId: r.trialId,
        effectiveConfig: r.effectiveConfig,
        transport: r.effectiveConfig.transport,
      };
    }
    case "transport_status": {
      const prev = state.transportStatus;
      let next: SessionViewState = { ...state, transportStatus: action.status, transportDetail: action.detail ?? null };
      if (action.status === "reconnecting" && prev !== "reconnecting") {
        next = localItem(next, {
          category: "error",
          source: "Transport",
          type: "gap",
          label: "Connection lost — reconnecting",
          detail: "Audio and events during the outage may not have been processed. The timeline will not interpolate missing state.",
          tone: "uncertain",
        });
      } else if (action.status === "connected" && prev === "reconnecting") {
        next = localItem(next, {
          category: "session",
          source: "Transport",
          label: "Connection restored",
          detail: action.detail ?? null,
          tone: "active",
        });
      }
      return next;
    }
    case "envelopes":
      return action.results.reduce(applyResult, state);
    case "local_error": {
      const e = action.error;
      const next: SessionViewState = {
        ...state,
        localError: e,
        phase: e.fatal ? "error" : state.phase,
        stopping: e.kind === "interrupt" ? false : state.stopping,
      };
      return e.fatal
        ? localItem(next, { category: "error", source: e.kind === "transport" ? "Transport" : "Session", label: e.message, detail: e.detail ?? null, tone: "danger" })
        : next;
    }
    case "end_requested":
      if (state.phase === "idle" || state.phase === "ended") return state;
      return localItem(
        { ...state, phase: "ending" },
        { category: "session", source: "Session", label: "End requested", detail: "Stopping microphone and playback", tone: "neutral" },
      );
    case "interrupt_requested":
      return localItem(
        { ...state, stopping: true },
        { category: "turn", source: "Session", label: "Interrupt requested", detail: "Manual interrupt — uses the barge-in cancellation path", tone: "danger" },
      );
    case "interrupt_failed":
      return {
        ...state,
        stopping: false,
        localError: { kind: "interrupt", message: action.message, retryable: true, fatal: false },
      };
    case "source_closed":
      return {
        ...state,
        phase: state.phase === "error" ? "error" : "ended",
        transportStatus: state.transportStatus === "error" ? "error" : "disconnected",
        // ephemeral live-only state is reset; the timeline and summary stay until cleared
        vad: null,
        audioLevel: { user: null, agent: null },
        playing: false,
        stopping: false,
      };
    case "clear_view": {
      const busy = isSessionBusy(state);
      if (!busy) return initialSessionState();
      return { ...state, timeline: [], timelineTrimmed: 0, transcript: [], errors: [], trialSummary: state.trialSummary };
    }
    case "dismiss_error": {
      const busy = isSessionBusy(state);
      return { ...state, localError: null, phase: state.phase === "error" && !busy ? "idle" : state.phase };
    }
    default:
      return state;
  }
}
