/**
 * Runtime validation at the API / event boundary (frontend.md §8).
 *
 * The wire format is snake_case (backend/src/turnsense/events.py, docs/api-contract.md).
 * Every object schema here validates the wire shape and converts its own keys to
 * camelCase. Dynamic identifier maps (latency metric names, comparison metric names,
 * group keys, tolerance names, trace `detail`) keep their wire keys because they are
 * data, not structure.
 */
import { z } from "zod";

// ---------------------------------------------------------------------------
// snake_case -> camelCase helpers
// ---------------------------------------------------------------------------

export type SnakeToCamel<S extends string> = S extends `${infer Head}_${infer Tail}`
  ? `${Head}${Capitalize<SnakeToCamel<Tail>>}`
  : S;

export type CamelizeKeys<T> = {
  [K in keyof T as K extends string ? SnakeToCamel<K> : K]: T[K];
};

export function snakeToCamel(key: string): string {
  return key.replace(/_([a-z0-9])/g, (_m, c: string) => c.toUpperCase());
}

export function camelToSnake(key: string): string {
  return key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`);
}

export function camelizeKeys<T extends object>(obj: T): CamelizeKeys<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) out[snakeToCamel(k)] = v;
  return out as CamelizeKeys<T>;
}

/** z.object(shape) whose parsed output has camelCase keys. */
function camelObject<S extends z.ZodRawShape>(shape: S) {
  return z
    .object(shape)
    .transform((o) => camelizeKeys(o as z.output<z.ZodObject<S>>) as CamelizeKeys<z.output<z.ZodObject<S>>>);
}

/** Accept null or a missing key; always output null when absent. */
function nul<T extends z.ZodTypeAny>(schema: T) {
  return schema.nullish().transform((v): z.output<T> | null => (v === undefined ? null : v));
}

// ---------------------------------------------------------------------------
// Enumerations (mirror events.py)
// ---------------------------------------------------------------------------

export const DETECTOR_MODES = ["silence_baseline", "learned_control", "anti_transfer"] as const;
export const TURN_EVENT_TYPES = [
  "CONTINUE",
  "POSSIBLE_END",
  "TURN_COMPLETE",
  "BARGE_IN",
  "BACKCHANNEL",
] as const;
export const DECISION_TYPES = ["TURN_COMPLETE", "BARGE_IN", "BACKCHANNEL_IGNORED"] as const;
export const SERVER_STATES = [
  "connecting",
  "listening",
  "user_speaking",
  "possible_end",
  "generating",
  "agent_speaking",
  "interrupted",
  "reconnecting",
  "error",
  "ending",
  "ended",
] as const;
export const TRANSPORTS = ["simulation", "livekit"] as const;
export const TRIAL_TYPES = ["simulated", "replay", "live"] as const;
export const TRACE_STATUSES = ["ok", "error", "cancelled", "timeout"] as const;
export const ERROR_SOURCES = ["transport", "stt", "llm", "tts", "detector", "vad", "worker"] as const;
export const LATENCY_METRICS = [
  "detector_inference",
  "turn_end_delay",
  "response_start",
  "barge_in_cancel",
  "stt_final",
  "llm_first_token",
  "tts_first_audio",
] as const;
export const ENVELOPE_KINDS = [
  "hello",
  "session_status",
  "vad",
  "turn_prediction",
  "turn_decision",
  "transcript",
  "trace",
  "latency",
  "latency_summary",
  "audio_level",
  "playback",
  "trial_summary",
  "error",
] as const;

export const PROTOCOL_VERSION = 1;

export const detectorModeSchema = z.enum(DETECTOR_MODES);
export const turnEventTypeSchema = z.enum(TURN_EVENT_TYPES);
export const decisionTypeSchema = z.enum(DECISION_TYPES);
export const serverStateSchema = z.enum(SERVER_STATES);
export const transportSchema = z.enum(TRANSPORTS);
export const trialTypeSchema = z.enum(TRIAL_TYPES);
export const traceStatusSchema = z.enum(TRACE_STATUSES);
export const errorSourceSchema = z.enum(ERROR_SOURCES);
export const latencyMetricSchema = z.enum(LATENCY_METRICS);

export type DetectorMode = z.infer<typeof detectorModeSchema>;
export type TurnEventType = z.infer<typeof turnEventTypeSchema>;
export type DecisionType = z.infer<typeof decisionTypeSchema>;
export type ServerState = z.infer<typeof serverStateSchema>;
export type Transport = z.infer<typeof transportSchema>;
export type TrialType = z.infer<typeof trialTypeSchema>;
export type TraceStatus = z.infer<typeof traceStatusSchema>;
export type ErrorSource = z.infer<typeof errorSourceSchema>;
export type LatencyMetric = z.infer<typeof latencyMetricSchema>;
export type EnvelopeKind = (typeof ENVELOPE_KINDS)[number];

export function isLatencyMetric(v: string): v is LatencyMetric {
  return (LATENCY_METRICS as readonly string[]).includes(v);
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

export const policyConfigSchema = camelObject({
  version: z.string(),
  silence_timeout_ms: z.number(),
  possible_end_ms: z.number(),
  complete_threshold: z.number(),
  possible_end_threshold: z.number(),
  consecutive_frames: z.number(),
  min_silence_guard_ms: z.number(),
  max_silence_ms: z.number(),
  barge_in_min_speech_ms: z.number(),
  backchannel_max_ms: z.number(),
  min_speech_ms: z.number(),
});
export type PolicyConfig = z.output<typeof policyConfigSchema>;

export const vadConfigSchema = camelObject({
  kind: z.string(),
  threshold: z.number(),
  min_speech_frames: z.number(),
  min_silence_frames: z.number(),
});

export const providerInfoSchema = camelObject({
  stt: z.string(),
  llm: z.string(),
  tts: z.string(),
});
export type ProviderInfo = z.output<typeof providerInfoSchema>;

export const effectiveConfigSchema = camelObject({
  detector_mode: detectorModeSchema,
  model_version: z.string(),
  decision_policy_version: z.string(),
  policy: policyConfigSchema,
  vad: vadConfigSchema,
  frame_ms: z.number(),
  sample_rate: z.number(),
  providers: providerInfoSchema,
  runtime_profile: z.string(),
  transport: transportSchema,
  scenario_id: z.string(),
  seed: nul(z.number()),
  transcript_enabled: z.boolean(),
  transcript_export_enabled: z.boolean(),
  raw_audio_retention: z.boolean(),
  speculative_prefill: z.boolean().default(false),
  server_version: z.string(),
  config_hash: z.string(),
});
export type EffectiveConfig = z.output<typeof effectiveConfigSchema>;

// ---------------------------------------------------------------------------
// Envelope payloads
// ---------------------------------------------------------------------------

export const sessionInfoSchema = camelObject({
  session_id: z.string(),
  trial_id: z.string(),
  transport: transportSchema,
  trial_type: trialTypeSchema,
  scenario_id: z.string(),
  trial_label: nul(z.string()),
  effective_config: effectiveConfigSchema,
  started_at: z.string(),
});
export type SessionInfo = z.output<typeof sessionInfoSchema>;

export const helloPayloadSchema = camelObject({
  protocol_version: z.number(),
  server_version: z.string(),
  session: sessionInfoSchema,
});

export const sessionStatusPayloadSchema = camelObject({
  state: serverStateSchema,
  previous_state: nul(serverStateSchema),
  reason: nul(z.string()),
});

export const vadPayloadSchema = camelObject({
  speaking: z.boolean(),
  probability: nul(z.number()),
  frame_end_ms: z.number(),
});

export const turnEventSchema = camelObject({
  session_id: z.string(),
  sequence: z.number(),
  event_type: turnEventTypeSchema,
  detector_mode: detectorModeSchema,
  model_version: z.string(),
  event_end_ms: z.number(),
  inference_ms: nul(z.number()),
  confidence: nul(z.number()),
  decision_policy_version: z.string(),
  reason_codes: z.array(z.string()).default([]),
});
export type TurnEvent = z.output<typeof turnEventSchema>;

export const turnDecisionPayloadSchema = camelObject({
  decision: decisionTypeSchema,
  source_event_sequence: nul(z.number()),
  last_user_frame_ms: z.number(),
  committed_ms: z.number(),
  confidence: nul(z.number()),
  reason_codes: z.array(z.string()).default([]),
});
export type TurnDecision = z.output<typeof turnDecisionPayloadSchema>;

export const transcriptPayloadSchema = camelObject({
  segment_id: z.string(),
  role: z.enum(["user", "assistant"]),
  text: z.string(),
  is_final: z.boolean(),
  revision: z.number(),
  start_ms: nul(z.number()),
  end_ms: nul(z.number()),
});
export type TranscriptPayload = z.output<typeof transcriptPayloadSchema>;

export const traceEventSchema = camelObject({
  session_id: z.string(),
  trial_id: nul(z.string()),
  trace_id: z.string(),
  stage: z.string(),
  timestamp_monotonic_ms: z.number(),
  duration_ms: nul(z.number()),
  detector_mode: nul(detectorModeSchema),
  provider: nul(z.string()),
  status: traceStatusSchema.default("ok"),
  generation_id: nul(z.number()),
  detail: nul(z.record(z.unknown())),
});
export type TraceEvent = z.output<typeof traceEventSchema>;

export const latencyPayloadSchema = camelObject({
  metric: latencyMetricSchema,
  value_ms: nul(z.number()),
  status: traceStatusSchema.default("ok"),
  generation_id: nul(z.number()),
});
export type LatencyPayload = z.output<typeof latencyPayloadSchema>;

export const percentileSummarySchema = camelObject({
  count: z.number(),
  failed: z.number().default(0),
  latest_ms: nul(z.number()),
  p50_ms: nul(z.number()),
  p95_ms: nul(z.number()),
  p99_ms: nul(z.number()),
  min_samples: z.number(),
});
export type PercentileSummary = z.output<typeof percentileSummarySchema>;

export const latencySummaryPayloadSchema = camelObject({
  // Keys are LatencyMetric wire values (identifiers, kept as-is). Unknown metric names
  // are dropped rather than failing the whole message.
  metrics: z.record(z.string(), percentileSummarySchema).transform((m) => {
    const out: Partial<Record<LatencyMetric, PercentileSummary>> = {};
    for (const [k, v] of Object.entries(m)) if (isLatencyMetric(k)) out[k] = v;
    return out;
  }),
});

export const audioLevelPayloadSchema = camelObject({
  source: z.enum(["user", "agent"]),
  rms_dbfs: z.number(),
});

export const playbackPayloadSchema = camelObject({
  action: z.enum(["start", "end", "flush"]),
  reason: nul(z.string()),
});

export const errorPayloadSchema = camelObject({
  source: errorSourceSchema,
  code: z.string(),
  message: z.string(),
  recoverable: z.boolean(),
});
export type ErrorPayload = z.output<typeof errorPayloadSchema>;

export const trialMetricsSchema = camelObject({
  turn_precision: nul(z.number()),
  turn_recall: nul(z.number()),
  turn_f1: nul(z.number()),
  premature_end_rate: nul(z.number()),
  missed_end_rate: nul(z.number()),
  turn_end_delay_p50_ms: nul(z.number()),
  turn_end_delay_p95_ms: nul(z.number()),
  response_start_p50_ms: nul(z.number()),
  response_start_p95_ms: nul(z.number()),
  response_start_p99_ms: nul(z.number()),
  detector_inference_p50_ms: nul(z.number()),
  detector_inference_p95_ms: nul(z.number()),
  detector_inference_p99_ms: nul(z.number()),
  false_barge_in_rate: nul(z.number()),
  missed_barge_in_rate: nul(z.number()),
  barge_in_cancel_p50_ms: nul(z.number()),
  stale_chunks_played: z.number().default(0),
  stale_chunks_rejected: z.number().default(0),
  dropped_frames: z.number().default(0),
  frame_trace_coverage: nul(z.number()),
});
export type TrialMetrics = z.output<typeof trialMetricsSchema>;

export const trialCountsSchema = camelObject({
  frames: z.number().default(0),
  reference_turn_ends: z.number().default(0),
  committed_turn_ends: z.number().default(0),
  reference_barge_ins: z.number().default(0),
  committed_barge_ins: z.number().default(0),
  agent_speaking_intervals: z.number().default(0),
  responses_started: z.number().default(0),
  responses_cancelled: z.number().default(0),
  errors: z.number().default(0),
});
export type TrialCounts = z.output<typeof trialCountsSchema>;

const toleranceSchema = nul(z.record(z.string(), z.number()));

export const trialSummaryPayloadSchema = camelObject({
  trial_id: z.string(),
  session_id: z.string(),
  trial_type: trialTypeSchema,
  scenario_id: z.string(),
  detector_mode: detectorModeSchema,
  model_version: z.string(),
  decision_policy_version: z.string(),
  labelled: z.boolean(),
  status: z.enum(["completed", "aborted", "error"]),
  metrics: trialMetricsSchema,
  counts: trialCountsSchema,
  tolerance_ms: toleranceSchema,
  notes: z.array(z.string()).default([]),
});
export type TrialSummary = z.output<typeof trialSummaryPayloadSchema>;

export const PAYLOAD_SCHEMAS = {
  hello: helloPayloadSchema,
  session_status: sessionStatusPayloadSchema,
  vad: vadPayloadSchema,
  turn_prediction: turnEventSchema,
  turn_decision: turnDecisionPayloadSchema,
  transcript: transcriptPayloadSchema,
  trace: traceEventSchema,
  latency: latencyPayloadSchema,
  latency_summary: latencySummaryPayloadSchema,
  audio_level: audioLevelPayloadSchema,
  playback: playbackPayloadSchema,
  trial_summary: trialSummaryPayloadSchema,
  error: errorPayloadSchema,
} as const satisfies Record<EnvelopeKind, z.ZodTypeAny>;

export type PayloadByKind = { [K in EnvelopeKind]: z.output<(typeof PAYLOAD_SCHEMAS)[K]> };

export interface EnvelopeBase {
  v: number;
  sessionId: string;
  seq: number;
  tMs: number;
  wallTime: string;
  generationId: number | null;
}

export type Envelope = {
  [K in EnvelopeKind]: EnvelopeBase & { kind: K; payload: PayloadByKind[K] };
}[EnvelopeKind];

export type EnvelopeOf<K extends EnvelopeKind> = Extract<Envelope, { kind: K }>;

const rawEnvelopeSchema = z.object({
  v: z.number().int(),
  kind: z.string(),
  session_id: z.string().min(1),
  seq: z.number().int().nonnegative(),
  t_ms: z.number(),
  wall_time: z.string(),
  generation_id: z.number().int().nullish(),
  payload: z.record(z.unknown()),
});

export type ParseFailureReason = "malformed" | "unknown_kind" | "invalid_payload" | "unsupported_version";

export type ParseResult =
  | { ok: true; envelope: Envelope; raw: unknown }
  | { ok: false; reason: ParseFailureReason; kind?: string; message: string; raw: unknown };

function isEnvelopeKind(kind: string): kind is EnvelopeKind {
  return (ENVELOPE_KINDS as readonly string[]).includes(kind);
}

/**
 * Validate one incoming message (a JSON string, bytes already decoded to a string, or an
 * object) and normalise it. Never throws: unknown kinds and invalid payloads are reported
 * so the caller can count them in diagnostics and carry on (frontend.md §8).
 */
export function parseEnvelope(input: unknown): ParseResult {
  let raw: unknown = input;
  if (typeof input === "string") {
    try {
      raw = JSON.parse(input);
    } catch {
      return { ok: false, reason: "malformed", message: "Message is not valid JSON", raw: input };
    }
  }
  const base = rawEnvelopeSchema.safeParse(raw);
  if (!base.success) {
    return {
      ok: false,
      reason: "malformed",
      message: base.error.issues[0]?.message ?? "Envelope did not match the contract",
      raw,
    };
  }
  const b = base.data;
  if (b.v !== PROTOCOL_VERSION) {
    return {
      ok: false,
      reason: "unsupported_version",
      kind: b.kind,
      message: `Envelope protocol v${b.v} is not supported (expected v${PROTOCOL_VERSION})`,
      raw,
    };
  }
  if (!isEnvelopeKind(b.kind)) {
    return { ok: false, reason: "unknown_kind", kind: b.kind, message: `Unknown kind "${b.kind}"`, raw };
  }
  const payload = PAYLOAD_SCHEMAS[b.kind].safeParse(b.payload);
  if (!payload.success) {
    const issue = payload.error.issues[0];
    return {
      ok: false,
      reason: "invalid_payload",
      kind: b.kind,
      message: issue ? `${issue.path.join(".") || "payload"}: ${issue.message}` : "Invalid payload",
      raw,
    };
  }
  const envelope = {
    v: b.v,
    kind: b.kind,
    sessionId: b.session_id,
    seq: b.seq,
    tMs: b.t_ms,
    wallTime: b.wall_time,
    generationId: b.generation_id ?? null,
    payload: payload.data,
  } as Envelope;
  return { ok: true, envelope, raw };
}

// ---------------------------------------------------------------------------
// REST responses
// ---------------------------------------------------------------------------

export const healthSchema = camelObject({
  status: z.string(),
  server_version: z.string(),
  protocol_version: z.number(),
  uptime_s: nul(z.number()),
});
export type Health = z.output<typeof healthSchema>;

export const detectorCapabilitySchema = camelObject({
  mode: detectorModeSchema,
  label: z.string(),
  model_version: z.string(),
  available: z.boolean(),
  unavailable_reason: nul(z.string()),
  probabilistic: z.boolean(),
});
export type DetectorCapability = z.output<typeof detectorCapabilitySchema>;

export const scenarioSchema = camelObject({
  id: z.string(),
  label: z.string(),
  description: z.string().default(""),
  trial_type: trialTypeSchema,
  labelled: z.boolean(),
  transports: z.array(transportSchema),
});
export type Scenario = z.output<typeof scenarioSchema>;

const transportAvailabilitySchema = camelObject({
  available: z.boolean(),
  reason: nul(z.string()),
});
export type TransportAvailability = z.output<typeof transportAvailabilitySchema>;

const UNAVAILABLE: TransportAvailability = { available: false, reason: "Not reported by server" };

export const capabilitiesSchema = camelObject({
  server_version: z.string(),
  protocol_version: z.number(),
  detectors: z.array(detectorCapabilitySchema),
  scenarios: z.array(scenarioSchema),
  transports: z
    .object({
      simulation: transportAvailabilitySchema.optional(),
      livekit: transportAvailabilitySchema.optional(),
    })
    .transform((t) => ({ simulation: t.simulation ?? UNAVAILABLE, livekit: t.livekit ?? UNAVAILABLE })),
  policy_defaults: policyConfigSchema,
  // Wire keys (policy field names) -> [min, max]. Non-tuple entries are ignored.
  policy_limits: z
    .record(z.string(), z.unknown())
    .default({})
    .transform((rec) => {
      const out: Record<string, [number, number]> = {};
      for (const [k, v] of Object.entries(rec)) {
        if (Array.isArray(v) && v.length === 2 && typeof v[0] === "number" && typeof v[1] === "number") {
          out[k] = [v[0], v[1]];
        }
      }
      return out;
    }),
  privacy: camelObject({
    raw_audio_retention: z.boolean(),
    transcript_display_default: z.boolean(),
    transcript_export_default: z.boolean(),
  }),
  runtime_profile: z.string(),
  providers: providerInfoSchema,
  min_percentile_samples: z.number().default(20),
  max_evaluation_repeats: z.number().default(50),
});
export type Capabilities = z.output<typeof capabilitiesSchema>;

export const liveKitJoinInfoSchema = camelObject({
  url: z.string(),
  room: z.string(),
  token: z.string(),
  expires_at: z.string(),
});
export type LiveKitJoinInfo = z.output<typeof liveKitJoinInfoSchema>;

export const createSessionResponseSchema = camelObject({
  session_id: z.string(),
  trial_id: z.string(),
  effective_config: effectiveConfigSchema,
  events_url: nul(z.string()),
  livekit: nul(liveKitJoinInfoSchema),
});
export type CreateSessionResponse = z.output<typeof createSessionResponseSchema>;

export const acceptedSchema = camelObject({ accepted: z.boolean() });

export const evaluationStatusSchema = camelObject({
  evaluation_id: z.string(),
  status: z.enum(["queued", "running", "completed", "stopped", "error"]),
  total: z.number(),
  completed: z.number(),
  trial_ids: z.array(z.string()),
  error: nul(z.string()),
  created_at: z.string(),
});
export type EvaluationStatus = z.output<typeof evaluationStatusSchema>;

const trialListItemShape = {
  trial_id: z.string(),
  session_id: z.string(),
  evaluation_id: nul(z.string()),
  created_at: z.string(),
  ended_at: nul(z.string()),
  trial_type: trialTypeSchema,
  transport: transportSchema,
  scenario_id: z.string(),
  trial_label: nul(z.string()),
  detector_mode: detectorModeSchema,
  model_version: z.string(),
  decision_policy_version: z.string(),
  // Display only; a running trial may report a status outside the summary enum.
  status: z.string(),
  labelled: z.boolean(),
  metrics: trialMetricsSchema,
  counts: trialCountsSchema,
};

export const trialListItemSchema = camelObject(trialListItemShape);
export type TrialListItem = z.output<typeof trialListItemSchema>;

export const trialListSchema = camelObject({
  total: z.number(),
  items: z.array(trialListItemSchema),
});
export type TrialList = z.output<typeof trialListSchema>;

export const annotationsSchema = camelObject({
  turn_ends_ms: z.array(z.number()).default([]),
  barge_ins_ms: z.array(z.number()).default([]),
  backchannels_ms: z.array(z.number()).default([]),
  source: nul(z.string()),
});
export type Annotations = z.output<typeof annotationsSchema>;

export const trialDetailSchema = camelObject({
  ...trialListItemShape,
  effective_config: effectiveConfigSchema,
  annotations: nul(annotationsSchema),
  tolerance_ms: toleranceSchema,
  events: z.array(z.unknown()).default([]),
}).transform((d) => {
  const parsed = d.events.map((e) => parseEnvelope(e));
  return {
    ...d,
    events: parsed.flatMap((p) => (p.ok ? [p.envelope] : [])),
    invalidEvents: parsed.filter((p) => !p.ok).length,
  };
});
export type TrialDetail = z.output<typeof trialDetailSchema>;

export const metricStatSchema = camelObject({
  mean: nul(z.number()),
  ci95_low: nul(z.number()),
  ci95_high: nul(z.number()),
  n: z.number(),
});
export type MetricStat = z.output<typeof metricStatSchema>;

export const comparisonGroupSchema = camelObject({
  // Group key: wire field name -> value. Kept as-is (identifiers).
  key: z.record(z.string(), z.union([z.string(), z.number(), z.null()])),
  n_trials: z.number(),
  scenario_ids: z.array(z.string()).default([]),
  trial_types: z.array(z.string()).default([]),
  metrics: z.record(z.string(), metricStatSchema),
});
export type ComparisonGroup = z.output<typeof comparisonGroupSchema>;

export const metricDefinitionSchema = camelObject({
  label: z.string(),
  better: z.string(),
  unit: z.string(),
  definition: z.string().default(""),
});
export type MetricDefinition = z.output<typeof metricDefinitionSchema>;

export const comparisonResultSchema = camelObject({
  group_by: z.array(z.string()),
  groups: z.array(comparisonGroupSchema),
  metric_definitions: z.record(z.string(), metricDefinitionSchema).default({}),
  evaluation_set: nul(z.string()),
  warnings: z.array(z.string()).default([]),
});
export type ComparisonResult = z.output<typeof comparisonResultSchema>;
