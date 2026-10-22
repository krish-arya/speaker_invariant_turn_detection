import { describe, expect, it } from "vitest";
import { MOCK_CAPABILITIES_WIRE } from "../events/mock";
import { decision, hello, prediction, SESSION, wire } from "../test/fixtures";
import {
  camelToSnake,
  capabilitiesSchema,
  comparisonResultSchema,
  parseEnvelope,
  snakeToCamel,
  trialSummaryPayloadSchema,
} from "./schemas";

describe("key conversion", () => {
  it("converts snake_case to camelCase including digits", () => {
    expect(snakeToCamel("turn_end_delay_p50_ms")).toBe("turnEndDelayP50Ms");
    expect(snakeToCamel("ci95_low")).toBe("ci95Low");
    expect(camelToSnake("turnEndDelayP50Ms")).toBe("turn_end_delay_p50_ms");
  });
});

describe("parseEnvelope", () => {
  it("validates and normalises a prediction envelope to camelCase", () => {
    const r = parseEnvelope(JSON.stringify(prediction("POSSIBLE_END", 0.72, 5)));
    expect(r.ok).toBe(true);
    if (!r.ok || r.envelope.kind !== "turn_prediction") throw new Error("expected turn_prediction");
    expect(r.envelope.sessionId).toBe(SESSION);
    expect(r.envelope.tMs).toBe(50);
    expect(r.envelope.payload.eventType).toBe("POSSIBLE_END");
    expect(r.envelope.payload.confidence).toBe(0.72);
    expect(r.envelope.payload.decisionPolicyVersion).toBe("policy-0.1");
    expect(r.envelope.payload.reasonCodes).toEqual(["stable_pause"]);
  });

  it("keeps null confidence for deterministic detectors (never fabricated)", () => {
    const r = parseEnvelope(prediction("TURN_COMPLETE", null, 2));
    expect(r.ok && r.envelope.kind === "turn_prediction" && r.envelope.payload.confidence).toBeNull();
  });

  it("normalises nested hello payload including effective config and policy", () => {
    const r = parseEnvelope(hello());
    if (!r.ok || r.envelope.kind !== "hello") throw new Error("expected hello");
    const cfg = r.envelope.payload.session.effectiveConfig;
    expect(cfg.detectorMode).toBe("anti_transfer");
    expect(cfg.policy.silenceTimeoutMs).toBe(700);
    expect(cfg.policy.bargeInMinSpeechMs).toBe(200);
    expect(cfg.rawAudioRetention).toBe(false);
  });

  it("parses a committed decision with its generation id", () => {
    const r = parseEnvelope(decision("BARGE_IN", 9, 4));
    if (!r.ok || r.envelope.kind !== "turn_decision") throw new Error("expected turn_decision");
    expect(r.envelope.generationId).toBe(4);
    expect(r.envelope.payload.lastUserFrameMs).toBe(2000);
  });

  it("reports unknown kinds without throwing", () => {
    const r = parseEnvelope(wire("future_thing", { a: 1 }, 3));
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.reason).toBe("unknown_kind");
      expect(r.kind).toBe("future_thing");
    }
  });

  it("reports invalid payloads and malformed JSON", () => {
    const bad = parseEnvelope(wire("vad", { speaking: "yes" }, 3));
    expect(bad.ok).toBe(false);
    if (!bad.ok) expect(bad.reason).toBe("invalid_payload");
    const junk = parseEnvelope("{not json");
    expect(junk.ok).toBe(false);
    if (!junk.ok) expect(junk.reason).toBe("malformed");
    const noSeq = parseEnvelope({ v: 1, kind: "vad", session_id: "s", t_ms: 1, wall_time: "x", payload: {} });
    expect(noSeq.ok).toBe(false);
  });

  it("rejects unsupported protocol versions", () => {
    const r = parseEnvelope({ ...wire("vad", { speaking: true, probability: null, frame_end_ms: 1 }, 1), v: 2 });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("unsupported_version");
  });

  it("drops unknown latency metric keys inside a summary instead of failing", () => {
    const r = parseEnvelope(
      wire(
        "latency_summary",
        {
          metrics: {
            response_start: { count: 3, failed: 1, latest_ms: 900, p50_ms: null, p95_ms: null, p99_ms: null, min_samples: 20 },
            something_new: { count: 1, failed: 0, latest_ms: 1, p50_ms: null, p95_ms: null, p99_ms: null, min_samples: 20 },
          },
        },
        4,
      ),
    );
    if (!r.ok || r.envelope.kind !== "latency_summary") throw new Error("expected latency_summary");
    expect(Object.keys(r.envelope.payload.metrics)).toEqual(["response_start"]);
    expect(r.envelope.payload.metrics.response_start?.failed).toBe(1);
    expect(r.envelope.payload.metrics.response_start?.p50Ms).toBeNull();
  });
});

describe("REST schemas", () => {
  it("parses capabilities and keeps policy limits keyed by wire names", () => {
    const caps = capabilitiesSchema.parse(MOCK_CAPABILITIES_WIRE);
    expect(caps.detectors[0]?.modelVersion).toBeTruthy();
    expect(caps.policyDefaults.silenceTimeoutMs).toBe(700);
    expect(caps.policyLimits.silence_timeout_ms).toEqual([50, 5000]);
    expect(caps.transports.livekit.available).toBe(false);
    expect(caps.privacy.rawAudioRetention).toBe(false);
  });

  it("ignores placeholder entries in policy_limits", () => {
    const caps = capabilitiesSchema.parse({ ...MOCK_CAPABILITIES_WIRE, policy_limits: { silence_timeout_ms: [50, 5000], "...": "..." } });
    expect(Object.keys(caps.policyLimits)).toEqual(["silence_timeout_ms"]);
  });

  it("camelises trial metrics but keeps nulls", () => {
    const s = trialSummaryPayloadSchema.parse({
      trial_id: "t",
      session_id: "s",
      trial_type: "live",
      scenario_id: "free_conversation",
      detector_mode: "silence_baseline",
      model_version: "m",
      decision_policy_version: "p",
      labelled: false,
      status: "completed",
      metrics: { turn_f1: null, response_start_p95_ms: 1200, stale_chunks_played: 0 },
      counts: { frames: 100 },
      tolerance_ms: null,
      notes: [],
    });
    expect(s.metrics.turnF1).toBeNull();
    expect(s.metrics.responseStartP95Ms).toBe(1200);
    expect(s.metrics.turnEndDelayP50Ms).toBeNull();
    expect(s.counts.frames).toBe(100);
    expect(s.counts.errors).toBe(0);
  });

  it("parses a comparison result with dynamic metric keys", () => {
    const r = comparisonResultSchema.parse({
      group_by: ["detector_mode"],
      groups: [
        {
          key: { detector_mode: "silence_baseline" },
          n_trials: 2,
          scenario_ids: ["barge_in"],
          trial_types: ["simulated"],
          metrics: { turn_f1: { mean: 0.8, ci95_low: null, ci95_high: null, n: 2 } },
        },
      ],
      metric_definitions: { turn_f1: { label: "Turn F1", better: "higher", unit: "ratio", definition: "…" } },
      evaluation_set: "sim-scenarios-v1",
      warnings: [],
    });
    expect(r.groups[0]?.metrics.turn_f1?.ci95Low).toBeNull();
    expect(r.metricDefinitions.turn_f1?.better).toBe("higher");
    expect(r.groups[0]?.key.detector_mode).toBe("silence_baseline");
  });
});
