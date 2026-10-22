/**
 * MOCK DATA ONLY. A scripted, realistic event stream for developing the console
 * without a backend (frontend.md §14 step 2). It emits wire-format (snake_case)
 * envelopes so the same validation path is exercised. Everything it produces is
 * labelled as mock in the UI and in the trial summary notes — never a measurement.
 */
import {
  capabilitiesSchema,
  createSessionResponseSchema,
  type Capabilities,
  type CreateSessionResponse,
  type DetectorMode,
  type EnvelopeKind,
  type LatencyMetric,
  type ServerState,
  type TurnEventType,
} from "../api/schemas";
import { POLICY_FIELDS } from "../lib/policy";
import { percentile, seededRandom } from "../lib/stats";
import type { SessionEventSource, SourceHandlers } from "./types";

export const MOCK_NOTE = "MOCK DATA — generated in the browser for UI development. These are not measurements.";

const POLICY_DEFAULTS_WIRE = {
  version: "policy-0.1",
  silence_timeout_ms: 700,
  possible_end_ms: 150,
  complete_threshold: 0.6,
  possible_end_threshold: 0.3,
  consecutive_frames: 3,
  min_silence_guard_ms: 100,
  max_silence_ms: 2000,
  barge_in_min_speech_ms: 200,
  backchannel_max_ms: 600,
  min_speech_ms: 60,
};

const POLICY_LIMITS_WIRE = {
  silence_timeout_ms: [50, 5000],
  possible_end_ms: [0, 5000],
  complete_threshold: [0, 1],
  possible_end_threshold: [0, 1],
  consecutive_frames: [1, 100],
  min_silence_guard_ms: [0, 5000],
  max_silence_ms: [100, 10000],
  barge_in_min_speech_ms: [0, 3000],
  backchannel_max_ms: [0, 3000],
  min_speech_ms: [0, 1000],
};

const MOCK_MODEL_VERSIONS: Record<DetectorMode, string> = {
  silence_baseline: "mock-silence-baseline-0.1",
  learned_control: "mock-learned-control-0.1",
  anti_transfer: "mock-anti-transfer-0.1",
};

export const MOCK_CAPABILITIES_WIRE = {
  server_version: "mock",
  protocol_version: 1,
  detectors: [
    { mode: "silence_baseline", label: "Silence/VAD baseline", model_version: MOCK_MODEL_VERSIONS.silence_baseline, available: true, unavailable_reason: null, probabilistic: false },
    { mode: "learned_control", label: "Learned control", model_version: MOCK_MODEL_VERSIONS.learned_control, available: true, unavailable_reason: null, probabilistic: true },
    { mode: "anti_transfer", label: "Anti-transfer", model_version: MOCK_MODEL_VERSIONS.anti_transfer, available: true, unavailable_reason: null, probabilistic: true },
  ],
  scenarios: [
    { id: "pause_then_continue", label: "Pause then continue", description: "User pauses mid-turn, then continues.", trial_type: "simulated", labelled: true, transports: ["simulation"] },
    { id: "short_answer", label: "Short answer", description: "A brief one-phrase turn.", trial_type: "simulated", labelled: true, transports: ["simulation"] },
    { id: "multi_turn", label: "Multi-turn", description: "Three user turns with varied pauses.", trial_type: "simulated", labelled: true, transports: ["simulation"] },
    { id: "barge_in", label: "Barge-in", description: "User interrupts the agent mid-response.", trial_type: "simulated", labelled: true, transports: ["simulation"] },
    { id: "backchannel", label: "Backchannel", description: "Short acknowledgement while the agent speaks.", trial_type: "simulated", labelled: true, transports: ["simulation"] },
  ],
  transports: {
    simulation: { available: true, reason: null },
    livekit: { available: false, reason: "Mock mode: no LiveKit server" },
  },
  policy_defaults: POLICY_DEFAULTS_WIRE,
  policy_limits: POLICY_LIMITS_WIRE,
  privacy: { raw_audio_retention: false, transcript_display_default: true, transcript_export_default: false },
  runtime_profile: "simulation",
  providers: { stt: "mock_stt", llm: "mock_llm", tts: "mock_tts" },
  min_percentile_samples: 20,
  max_evaluation_repeats: 50,
};

export const MOCK_CAPABILITIES: Capabilities = capabilitiesSchema.parse(MOCK_CAPABILITIES_WIRE);

export interface MockSessionOptions {
  detectorMode: DetectorMode;
  scenarioId: string;
  policyOverrides: Record<string, number> | null;
  transcriptEnabled: boolean;
  transcriptExportEnabled: boolean;
  speculativePrefill?: boolean;
  trialLabel: string | null;
  seed: number | null;
  /** >1 runs the script faster (tests). */
  speed?: number;
}

type Wire = Record<string, unknown>;

interface Step {
  at: number;
  kind: EnvelopeKind;
  payload: Wire;
  gen: number | null;
  tag?: "agent";
}

const randomId = (prefix: string, rng: () => number) =>
  `${prefix}_${Math.floor(rng() * 0xffffffff).toString(16).padStart(8, "0")}${Math.floor(rng() * 0xffff).toString(16).padStart(4, "0")}`;

// ---------------------------------------------------------------------------
// Script construction
// ---------------------------------------------------------------------------

class ScriptBuilder {
  readonly steps: Step[] = [];
  gen = 0;
  detSeq = 0;
  segCounter = 0;
  readonly refEnds: number[] = [];
  readonly commits: number[] = [];
  readonly bargeIns: number[] = [];
  readonly backchannels: number[] = [];
  readonly learned: boolean;

  constructor(
    readonly sessionId: string,
    readonly mode: DetectorMode,
    readonly modelVersion: string,
    readonly policy: typeof POLICY_DEFAULTS_WIRE,
    readonly transcripts: boolean,
    readonly rng: () => number,
  ) {
    this.learned = mode !== "silence_baseline";
  }

  push(at: number, kind: EnvelopeKind, payload: Wire, gen: number | null = null, tag?: "agent"): void {
    this.steps.push({ at: Math.round(at * 10) / 10, kind, payload, gen, ...(tag ? { tag } : {}) });
  }

  status(at: number, state: ServerState, prev: ServerState | null, reason: string | null, gen: number | null = null, tag?: "agent"): void {
    this.push(at, "session_status", { state, previous_state: prev, reason }, gen, tag);
  }

  vad(at: number, speaking: boolean): void {
    this.push(at, "vad", { speaking, probability: null, frame_end_ms: at });
  }

  conf(learnedValue: number): number | null {
    return this.learned ? Math.min(0.99, Math.max(0.01, learnedValue + (this.rng() - 0.5) * 0.06)) : null;
  }

  predict(at: number, eventType: TurnEventType, confidence: number | null, reasons: string[], gen: number | null = null, tag?: "agent"): number {
    this.detSeq += 1;
    this.push(
      at,
      "turn_prediction",
      {
        session_id: this.sessionId,
        sequence: this.detSeq,
        event_type: eventType,
        detector_mode: this.mode,
        model_version: this.modelVersion,
        event_end_ms: at,
        inference_ms: this.learned ? 4 + this.rng() * 3 : 0.04,
        confidence,
        decision_policy_version: this.policy.version,
        reason_codes: reasons,
      },
      gen,
      tag,
    );
    return this.detSeq;
  }

  trace(at: number, stage: string, durationMs: number | null, gen: number | null, provider: string | null = null, status = "ok", tag?: "agent"): void {
    this.push(
      at,
      "trace",
      {
        session_id: this.sessionId,
        trial_id: null,
        trace_id: `trc_${stage}_${Math.round(at)}`,
        stage,
        timestamp_monotonic_ms: at,
        duration_ms: durationMs,
        detector_mode: this.mode,
        provider,
        status,
        generation_id: gen,
        detail: null,
      },
      gen,
      tag,
    );
  }

  latency(at: number, metric: LatencyMetric, value: number | null, gen: number | null, tag?: "agent"): void {
    this.push(at, "latency", { metric, value_ms: value, status: "ok", generation_id: gen }, gen, tag);
  }

  levels(from: number, to: number, source: "user" | "agent", base: number, gen: number | null = null, tag?: "agent"): void {
    for (let t = from; t < to; t += 100) {
      const v = source === "user" && base <= -55 ? -60 + this.rng() * 3 : base + (this.rng() - 0.5) * 12;
      this.push(t, "audio_level", { source, rms_dbfs: Math.min(-3, v) }, gen, tag);
    }
  }

  transcript(at: number, segmentId: string, role: "user" | "assistant", text: string, isFinal: boolean, revision: number, startMs: number | null, endMs: number | null, gen: number | null = null, tag?: "agent"): void {
    if (!this.transcripts) return;
    this.push(at, "transcript", { segment_id: segmentId, role, text, is_final: isFinal, revision, start_ms: startMs, end_ms: endMs }, gen, tag);
  }

  /** A user turn; returns the annotated end, the commit time and the new generation. */
  userTurn(start: number, segments: Array<{ speak: number; pause: number }>, text: string, prev: ServerState): { end: number; commit: number; gen: number } {
    const p = this.policy;
    let t = start;
    this.status(t, "user_speaking", prev, "VAD speech start");
    this.vad(t, true);
    this.predict(t + 40, "CONTINUE", this.conf(0.93), ["speech_active"]);
    const words = text.split(" ");
    const totalSpeak = segments.reduce((a, s) => a + s.speak, 0);
    const segId = `seg_u${++this.segCounter}`;
    let spoken = 0;
    let revision = 0;

    segments.forEach((seg, i) => {
      this.levels(t, t + seg.speak, "user", -18);
      for (let pt = t + 400; pt < t + seg.speak; pt += 450) {
        const frac = (spoken + (pt - t)) / totalSpeak;
        const n = Math.max(1, Math.round(words.length * frac));
        this.transcript(pt, segId, "user", words.slice(0, n).join(" "), false, ++revision, start, null);
      }
      spoken += seg.speak;
      t += seg.speak;
      this.vad(t, false);
      const last = i === segments.length - 1;
      if (!last) {
        if (seg.pause > p.possible_end_ms) {
          this.predict(t + p.possible_end_ms, "POSSIBLE_END", this.conf(0.38), this.learned ? ["pause", "low_turn_end_score"] : ["silence"]);
          this.status(t + p.possible_end_ms, "possible_end", "user_speaking", "pause detected");
        }
        this.levels(t, t + seg.pause, "user", -60);
        t += seg.pause;
        this.vad(t, true);
        this.status(t, "user_speaking", seg.pause > p.possible_end_ms ? "possible_end" : "user_speaking", "speech resumed");
        this.predict(t + 20, "CONTINUE", this.conf(0.9), ["speech_resumed"]);
      }
    });

    const end = t;
    this.predict(end + p.possible_end_ms, "POSSIBLE_END", this.conf(0.47), this.learned ? ["pause", "rising_turn_end_score"] : ["silence"]);
    this.status(end + p.possible_end_ms, "possible_end", "user_speaking", "pause detected");
    this.levels(end, end + 400, "user", -60);
    const sttDur = 160 + this.rng() * 60;
    this.transcript(end + sttDur, segId, "user", text, true, ++revision, start, end);
    this.trace(end + sttDur, "stt_final", sttDur, null, "mock_stt");
    this.latency(end + sttDur, "stt_final", sttDur, null);

    const frames = p.consecutive_frames * 20;
    const commitDelay = this.learned ? Math.max(p.min_silence_guard_ms, p.possible_end_ms + 110) + frames : p.silence_timeout_ms;
    const commit = end + commitDelay;
    const predSeq = this.predict(
      commit - (this.learned ? frames : 0),
      "TURN_COMPLETE",
      this.conf(0.86),
      this.learned ? ["stable_pause", "high_turn_end_score"] : ["silence_timeout"],
    );
    const gen = ++this.gen;
    this.push(
      commit,
      "turn_decision",
      {
        decision: "TURN_COMPLETE",
        source_event_sequence: predSeq,
        last_user_frame_ms: end,
        committed_ms: commit,
        confidence: this.conf(0.86),
        reason_codes: this.learned ? ["consecutive_frames_met", "min_silence_guard_met"] : ["silence_timeout"],
      },
      gen,
    );
    this.trace(commit, "turn_decision", null, gen);
    this.status(commit, "generating", "possible_end", "TURN_COMPLETE committed", gen);
    this.latency(commit, "turn_end_delay", commit - end, gen);
    this.refEnds.push(end);
    this.commits.push(commit);
    return { end, commit, gen };
  }

  /** Agent response for generation `gen`; returns playback start and end. */
  agentResponse(commit: number, lastUserFrame: number, gen: number, text: string, durationMs: number): { playStart: number; playEnd: number } {
    this.trace(commit + 3, "llm_request", null, gen, "mock_llm");
    const llm = 150 + this.rng() * 90;
    this.trace(commit + 3 + llm, "llm_first_token", llm, gen, "mock_llm");
    this.latency(commit + 3 + llm, "llm_first_token", llm, gen);
    const ttsReq = commit + 3 + llm + 55;
    this.trace(ttsReq, "tts_request", null, gen, "mock_tts");
    const tts = 70 + this.rng() * 60;
    this.trace(ttsReq + tts, "tts_first_audio", tts, gen, "mock_tts");
    this.latency(ttsReq + tts, "tts_first_audio", tts, gen);
    const playStart = ttsReq + tts + 6;
    const playEnd = playStart + durationMs;
    this.push(playStart, "playback", { action: "start", reason: null }, gen);
    this.status(playStart, "agent_speaking", "generating", "first TTS audio ready", gen);
    this.latency(playStart, "response_start", playStart - lastUserFrame, gen);

    const segId = `seg_a${gen}`;
    const words = text.split(" ");
    let rev = 0;
    for (let pt = playStart + 250; pt < playEnd; pt += 500) {
      const n = Math.max(1, Math.round((words.length * (pt - playStart)) / durationMs));
      this.transcript(pt, segId, "assistant", words.slice(0, n).join(" "), false, ++rev, playStart, null, gen, "agent");
    }
    this.levels(playStart, playEnd, "agent", -16, gen, "agent");
    this.trace(playStart + durationMs * 0.6, "llm_complete", durationMs * 0.6, gen, "mock_llm", "ok", "agent");
    this.transcript(playEnd, segId, "assistant", text, true, ++rev, playStart, playEnd, gen, "agent");
    this.push(playEnd, "playback", { action: "end", reason: "completed" }, gen, "agent");
    this.status(playEnd + 5, "listening", "agent_speaking", "playback completed", gen, "agent");
    return { playStart, playEnd };
  }

  /** Drop the still-pending agent output of `gen` at or after `from`. */
  cancelAgentAfter(gen: number, from: number): void {
    for (let i = this.steps.length - 1; i >= 0; i -= 1) {
      const s = this.steps[i]!;
      if (s.tag === "agent" && s.gen === gen && s.at >= from) this.steps.splice(i, 1);
    }
  }

  /** User barge-in during agent output. Returns when the user's turn continues. */
  bargeIn(tb: number, gen: number, manual: boolean): number {
    const p = this.policy;
    let commitB: number;
    if (manual) {
      commitB = tb;
    } else {
      this.vad(tb, true);
      this.levels(tb, tb + p.barge_in_min_speech_ms + 60, "user", -17);
      this.predict(tb + p.barge_in_min_speech_ms, "BARGE_IN", this.conf(0.89), ["speech_during_agent", "sustained_speech"], gen);
      commitB = tb + p.barge_in_min_speech_ms + 20;
      this.bargeIns.push(tb);
    }
    this.cancelAgentAfter(gen, commitB);
    this.push(
      commitB,
      "turn_decision",
      {
        decision: "BARGE_IN",
        source_event_sequence: manual ? null : this.detSeq,
        last_user_frame_ms: manual ? commitB : commitB - 5,
        committed_ms: commitB,
        confidence: manual ? null : this.conf(0.89),
        reason_codes: manual ? ["manual_interrupt"] : ["barge_in_min_speech_met"],
      },
      gen,
    );
    this.status(commitB, "interrupted", "agent_speaking", manual ? "manual interrupt" : "BARGE_IN committed", gen);
    this.trace(commitB + 1, "barge_in_detected", null, gen);
    this.trace(commitB + 2, "cancel_requested", null, gen);
    const flushAt = commitB + 28 + this.rng() * 20;
    this.push(flushAt, "playback", { action: "flush", reason: "barge-in" }, gen);
    this.trace(flushAt + 3, "cancel_acknowledged", flushAt + 3 - commitB, gen);
    this.latency(flushAt + 3, "barge_in_cancel", flushAt + 3 - commitB, gen);
    // A late chunk from the cancelled generation: the console must ignore it.
    this.transcript(flushAt + 30, `seg_a${gen}`, "assistant", "[late chunk from cancelled response]", false, 999, null, null, gen);
    this.trace(flushAt + 32, "stale_chunk_rejected", null, gen);
    const resume = flushAt + 8;
    this.status(resume, manual ? "listening" : "user_speaking", "interrupted", "cancel acknowledged");
    return resume;
  }

  backchannel(tb: number, gen: number): void {
    const dur = 160;
    this.vad(tb, true);
    this.levels(tb, tb + dur, "user", -24);
    const seq = this.predict(tb + 60, this.learned ? "BACKCHANNEL" : "CONTINUE", this.conf(0.74), ["short_speech_during_agent"], gen);
    this.vad(tb + dur, false);
    this.push(
      tb + dur + 20,
      "turn_decision",
      {
        decision: "BACKCHANNEL_IGNORED",
        source_event_sequence: seq,
        last_user_frame_ms: tb + dur,
        committed_ms: tb + dur + 20,
        confidence: this.conf(0.74),
        reason_codes: ["speech_shorter_than_barge_in_min"],
      },
      gen,
    );
    this.backchannels.push(tb);
  }
}

function buildScript(b: ScriptBuilder, scenarioId: string): { endAt: number | null } {
  b.status(300, "listening", "connecting", "room joined");
  const finish = (t: number) => ({ endAt: t + 900 });

  switch (scenarioId) {
    case "short_answer": {
      const u = b.userTurn(1200, [{ speak: 520, pause: 0 }], "Yes, please.", "listening");
      const a = b.agentResponse(u.commit, u.end, u.gen, "Great — I've booked it for you.", 1800);
      return finish(a.playEnd);
    }
    case "barge_in": {
      const u = b.userTurn(1200, [{ speak: 1600, pause: 0 }], "Can you tell me about the weather this weekend", "listening");
      const a = b.agentResponse(u.commit, u.end, u.gen, "This weekend looks mostly sunny on Saturday with highs around twenty-four degrees, then on Sunday clouds move in and there is a chance of light rain in the afternoon.", 5200);
      const resume = b.bargeIn(a.playStart + 1300, u.gen, false);
      const u2 = b.userTurn(resume, [{ speak: 1100, pause: 0 }], "Actually just Sunday morning", "user_speaking");
      const a2 = b.agentResponse(u2.commit, u2.end, u2.gen, "Sunday morning should be dry and mild.", 1900);
      return finish(a2.playEnd);
    }
    case "backchannel": {
      const u = b.userTurn(1200, [{ speak: 1400, pause: 0 }], "How do I reset my router", "listening");
      const a = b.agentResponse(u.commit, u.end, u.gen, "First unplug the router, wait about thirty seconds, then plug it back in and wait for the lights to settle before reconnecting.", 4200);
      b.backchannel(a.playStart + 1500, u.gen);
      return finish(a.playEnd);
    }
    case "multi_turn":
    case "free_conversation": {
      const u1 = b.userTurn(1200, [{ speak: 900, pause: 0 }], "Hi, I need a taxi", "listening");
      const a1 = b.agentResponse(u1.commit, u1.end, u1.gen, "Sure. Where would you like to go?", 1700);
      const u2 = b.userTurn(a1.playEnd + 900, [{ speak: 1100, pause: 600 }, { speak: 900, pause: 0 }], "To the central station — and, um, as soon as possible", "listening");
      const a2 = b.agentResponse(u2.commit, u2.end, u2.gen, "A taxi to the central station is on its way, arriving in about six minutes.", 2600);
      const u3 = b.userTurn(a2.playEnd + 1000, [{ speak: 600, pause: 0 }], "Thanks a lot", "listening");
      const a3 = b.agentResponse(u3.commit, u3.end, u3.gen, "You're welcome. Have a good trip!", 1500);
      return scenarioId === "free_conversation" ? { endAt: null } : finish(a3.playEnd);
    }
    case "pause_then_continue":
    default: {
      const u = b.userTurn(1200, [{ speak: 1300, pause: 550 }, { speak: 1400, pause: 0 }], "I'd like to book a table for two — at around seven tonight", "listening");
      const a = b.agentResponse(u.commit, u.end, u.gen, "Of course. A table for two at seven tonight. Any seating preference?", 2600);
      return finish(a.playEnd);
    }
  }
}

// ---------------------------------------------------------------------------
// Runtime
// ---------------------------------------------------------------------------

export class MockEventSource implements SessionEventSource {
  readonly kind = "mock" as const;
  readonly sessionId: string;
  readonly trialId: string;
  private readonly rng: () => number;
  private readonly builder: ScriptBuilder;
  private readonly configWire: Wire;
  private readonly labelled: boolean;
  private steps: Step[];
  private idx = 0;
  private seq = 0;
  private handlers: SourceHandlers | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private t0 = 0;
  private wall0 = Date.now();
  private lastSummaryAt = 0;
  private finished = false;
  private state: ServerState = "connecting";
  private activeGen: number | null = null;
  private readonly samples: Partial<Record<LatencyMetric, number[]>> = {};
  private cancelledCount = 0;
  private staleRejected = 0;
  private respStarted = 0;
  private agentIntervals = 0;
  private readonly speed: number;

  constructor(
    private readonly opts: MockSessionOptions,
    private readonly clock: () => number = () => performance.now(),
  ) {
    this.rng = seededRandom(opts.seed ?? 7);
    this.speed = opts.speed ?? 1;
    this.sessionId = randomId("mock_ses", this.rng);
    this.trialId = randomId("mock_tri", this.rng);
    const policy = { ...POLICY_DEFAULTS_WIRE };
    for (const f of POLICY_FIELDS) {
      const v = opts.policyOverrides?.[f.wire];
      if (typeof v === "number") (policy as Record<string, number | string>)[f.wire] = v;
    }
    const modelVersion = MOCK_MODEL_VERSIONS[opts.detectorMode];
    this.configWire = {
      detector_mode: opts.detectorMode,
      model_version: modelVersion,
      decision_policy_version: policy.version,
      policy,
      vad: { kind: "energy", threshold: 0.5, min_speech_frames: 2, min_silence_frames: 2 },
      frame_ms: 20,
      sample_rate: 16000,
      providers: { stt: "mock_stt", llm: "mock_llm", tts: "mock_tts" },
      runtime_profile: "simulation",
      transport: "simulation",
      scenario_id: opts.scenarioId,
      seed: opts.seed,
      transcript_enabled: opts.transcriptEnabled,
      transcript_export_enabled: opts.transcriptExportEnabled,
      raw_audio_retention: false,
      speculative_prefill: opts.speculativePrefill ?? false,
      server_version: "mock",
      config_hash: "mock0000",
    };
    this.labelled = opts.scenarioId !== "free_conversation";
    this.builder = new ScriptBuilder(this.sessionId, opts.detectorMode, modelVersion, policy, opts.transcriptEnabled, this.rng);
    const { endAt } = buildScript(this.builder, opts.scenarioId);
    if (endAt !== null) {
      this.builder.status(endAt, "ending", "listening", "scenario complete");
      this.builder.push(endAt + 60, "trial_summary", {}, null); // filled at emission time
      this.builder.status(endAt + 120, "ended", "ending", "scenario complete");
    }
    this.steps = this.builder.steps.slice().sort((a, b) => a.at - b.at);
  }

  createSessionResponse(): CreateSessionResponse {
    return createSessionResponseSchema.parse({
      session_id: this.sessionId,
      trial_id: this.trialId,
      effective_config: this.configWire,
      events_url: null,
      livekit: null,
    });
  }

  private now(): number {
    return (this.clock() - this.t0) * this.speed;
  }

  start(h: SourceHandlers): Promise<void> {
    this.handlers = h;
    h.onStatus("connecting");
    this.t0 = this.clock();
    this.wall0 = Date.now();
    this.emit({
      at: 0,
      kind: "hello",
      gen: null,
      payload: {
        protocol_version: 1,
        server_version: "mock",
        session: {
          session_id: this.sessionId,
          trial_id: this.trialId,
          transport: "simulation",
          trial_type: "simulated",
          scenario_id: this.opts.scenarioId,
          trial_label: this.opts.trialLabel,
          effective_config: this.configWire,
          started_at: new Date(this.wall0).toISOString(),
        },
      },
    });
    this.emit({ at: 1, kind: "session_status", gen: null, payload: { state: "connecting", previous_state: null, reason: "mock session" } });
    h.onStatus("connected");
    this.timer = setInterval(() => this.tick(), 40);
    return Promise.resolve();
  }

  /** Emit everything due up to now. Exposed for deterministic tests. */
  tick(nowOverride?: number): void {
    if (this.finished) return;
    const now = nowOverride ?? this.now();
    while (this.idx < this.steps.length && this.steps[this.idx]!.at <= now) {
      const step = this.steps[this.idx++]!;
      this.emit(step);
      if (this.finished) return;
    }
    if (now - this.lastSummaryAt >= 1000) {
      this.lastSummaryAt = now;
      this.emitPeriodic(now);
    }
  }

  private emitPeriodic(now: number): void {
    const det = this.builder.learned ? 4 + this.rng() * 3 : 0.04 + this.rng() * 0.02;
    this.emit({ at: now, kind: "latency", gen: null, payload: { metric: "detector_inference", value_ms: det, status: "ok", generation_id: null } });
    this.emit({ at: now, kind: "latency_summary", gen: null, payload: { metrics: this.summaryWire(now) } });
  }

  private detectorStats(): [number, number, number] {
    return this.builder.learned ? [4.6, 6.8, 8.9] : [0.04, 0.06, 0.09];
  }

  private summaryWire(now: number): Wire {
    const min = 20;
    const out: Wire = {};
    const [p50, p95, p99] = this.detectorStats();
    const detCount = Math.floor(now / 20);
    out.detector_inference = {
      count: detCount,
      failed: 0,
      latest_ms: p50,
      p50_ms: detCount >= min ? p50 : null,
      p95_ms: detCount >= min ? p95 : null,
      p99_ms: detCount >= min ? p99 : null,
      min_samples: min,
    };
    for (const metric of ["turn_end_delay", "response_start", "barge_in_cancel", "stt_final", "llm_first_token", "tts_first_audio"] as const) {
      const vals = this.samples[metric] ?? [];
      const enough = vals.length >= min;
      out[metric] = {
        count: vals.length,
        failed: 0,
        latest_ms: vals.length ? vals[vals.length - 1] : null,
        p50_ms: enough ? percentile(vals, 50) : null,
        p95_ms: enough ? percentile(vals, 95) : null,
        p99_ms: enough ? percentile(vals, 99) : null,
        min_samples: min,
      };
    }
    return out;
  }

  private emit(step: Step): void {
    const h = this.handlers;
    if (!h) return;
    let payload = step.payload;
    if (step.kind === "trial_summary") payload = this.trialSummaryWire();
    if (step.kind === "session_status") {
      this.state = payload.state as ServerState;
      if (this.state === "agent_speaking") this.agentIntervals += 1;
    }
    if (step.kind === "turn_decision") {
      if (payload.decision === "TURN_COMPLETE") this.activeGen = step.gen;
      if (payload.decision === "BARGE_IN") this.cancelledCount += 1;
    }
    if (step.kind === "playback" && payload.action === "start") this.respStarted += 1;
    if (step.kind === "trace" && payload.stage === "stale_chunk_rejected") this.staleRejected += 1;
    if (step.kind === "latency" && typeof payload.value_ms === "number") {
      const m = payload.metric as LatencyMetric;
      (this.samples[m] ??= []).push(payload.value_ms);
    }
    this.seq += 1;
    h.onMessage(
      JSON.stringify({
        v: 1,
        kind: step.kind,
        session_id: this.sessionId,
        seq: this.seq,
        t_ms: step.at,
        wall_time: new Date(this.wall0 + step.at / this.speed).toISOString(),
        generation_id: step.gen,
        payload,
      }),
    );
    if (step.kind === "session_status" && payload.state === "ended") this.finish();
  }

  private trialSummaryWire(): Wire {
    const b = this.builder;
    const delays = b.commits.map((c, i) => c - (b.refEnds[i] ?? c));
    const resp = this.samples.response_start ?? [];
    const cancel = this.samples.barge_in_cancel ?? [];
    const [d50, d95, d99] = this.detectorStats();
    const labelled = this.labelled;
    const matched = Math.min(b.commits.length, b.refEnds.length);
    const precision = b.commits.length ? matched / b.commits.length : null;
    const recall = b.refEnds.length ? matched / b.refEnds.length : null;
    const f1 = precision !== null && recall !== null && precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : null;
    return {
      trial_id: this.trialId,
      session_id: this.sessionId,
      trial_type: "simulated",
      scenario_id: this.opts.scenarioId,
      detector_mode: this.opts.detectorMode,
      model_version: this.configWire.model_version,
      decision_policy_version: this.configWire.decision_policy_version,
      labelled,
      status: "completed",
      metrics: {
        turn_precision: labelled ? precision : null,
        turn_recall: labelled ? recall : null,
        turn_f1: labelled ? f1 : null,
        premature_end_rate: labelled ? 0 : null,
        missed_end_rate: labelled ? 0 : null,
        turn_end_delay_p50_ms: labelled ? percentile(delays, 50) : null,
        turn_end_delay_p95_ms: labelled ? percentile(delays, 95) : null,
        response_start_p50_ms: percentile(resp, 50),
        response_start_p95_ms: percentile(resp, 95),
        response_start_p99_ms: percentile(resp, 99),
        detector_inference_p50_ms: d50,
        detector_inference_p95_ms: d95,
        detector_inference_p99_ms: d99,
        false_barge_in_rate: labelled && this.agentIntervals ? 0 : null,
        missed_barge_in_rate: labelled && b.bargeIns.length ? 0 : null,
        barge_in_cancel_p50_ms: percentile(cancel, 50),
        stale_chunks_played: 0,
        stale_chunks_rejected: this.staleRejected,
        dropped_frames: 0,
        frame_trace_coverage: 1,
      },
      counts: {
        frames: Math.floor(this.now() / 20),
        reference_turn_ends: labelled ? b.refEnds.length : 0,
        committed_turn_ends: b.commits.length,
        reference_barge_ins: labelled ? b.bargeIns.length : 0,
        committed_barge_ins: this.cancelledCount,
        agent_speaking_intervals: this.agentIntervals,
        responses_started: this.respStarted,
        responses_cancelled: this.cancelledCount,
        errors: 0,
      },
      tolerance_ms: labelled ? { early: 200, late: 1000, barge_in: 600 } : null,
      notes: [MOCK_NOTE],
    };
  }

  /** Mirrors POST /interrupt: only valid while a response is generating or playing. */
  interrupt(): Promise<void> {
    if (this.finished || (this.state !== "generating" && this.state !== "agent_speaking") || this.activeGen === null) {
      return Promise.reject(new Error("Nothing to interrupt right now."));
    }
    const now = this.now() + 5;
    // Re-run the barge-in builder over the pending steps only, then splice them back.
    this.builder.steps.length = 0;
    this.builder.steps.push(...this.steps.slice(this.idx));
    this.builder.bargeIn(now, this.activeGen, true);
    this.steps = this.steps.slice(0, this.idx).concat(this.builder.steps.slice().sort((a, b) => a.at - b.at));
    return Promise.resolve();
  }

  /** Mirrors POST /end: emits ending → trial_summary → ended, then closes. */
  end(): Promise<void> {
    if (this.finished) return Promise.resolve();
    const now = this.now();
    this.steps = this.steps.slice(0, this.idx);
    if (this.state !== "ending") {
      this.steps.push({ at: now, kind: "session_status", payload: { state: "ending", previous_state: this.state, reason: "user ended session" }, gen: null });
    }
    this.steps.push({ at: now + 1, kind: "trial_summary", payload: {}, gen: null });
    this.steps.push({ at: now + 2, kind: "session_status", payload: { state: "ended", previous_state: "ending", reason: "user ended session" }, gen: null });
    this.tick(now + 2);
    return Promise.resolve();
  }

  private finish(): void {
    if (this.finished) return;
    this.finished = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    const h = this.handlers;
    setTimeout(() => h?.onStatus("disconnected", "Mock session finished."), 30);
  }

  stop(): Promise<void> {
    const wasFinished = this.finished;
    this.finished = true;
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    if (!wasFinished) this.handlers?.onStatus("disconnected");
    return Promise.resolve();
  }
}
