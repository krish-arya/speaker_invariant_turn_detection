/** Wire-format (snake_case) fixtures matching backend/src/turnsense/events.py. */

export const SESSION = "ses_test0001";

export const effectiveConfigWire = (overrides: Record<string, unknown> = {}) => ({
  detector_mode: "anti_transfer",
  model_version: "turn-detector-at-0.1.0",
  decision_policy_version: "policy-0.1",
  policy: {
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
  },
  vad: { kind: "energy", threshold: 0.5, min_speech_frames: 2, min_silence_frames: 2 },
  frame_ms: 20,
  sample_rate: 16000,
  providers: { stt: "mock_stt", llm: "mock_llm", tts: "mock_tts" },
  runtime_profile: "simulation",
  transport: "simulation",
  scenario_id: "pause_then_continue",
  seed: 7,
  transcript_enabled: true,
  transcript_export_enabled: false,
  raw_audio_retention: false,
  speculative_prefill: false,
  server_version: "0.1.0",
  config_hash: "3f9a1c2e",
  ...overrides,
});

export function wire(
  kind: string,
  payload: Record<string, unknown>,
  seq: number,
  opts: { session?: string; t?: number; gen?: number | null } = {},
) {
  return {
    v: 1,
    kind,
    session_id: opts.session ?? SESSION,
    seq,
    t_ms: opts.t ?? seq * 10,
    wall_time: "2026-10-22T08:00:00.000Z",
    generation_id: opts.gen ?? null,
    payload,
  };
}

export const status = (state: string, prev: string | null, seq: number, gen: number | null = null) =>
  wire("session_status", { state, previous_state: prev, reason: null }, seq, { gen });

export const prediction = (eventType: string, confidence: number | null, seq: number) =>
  wire(
    "turn_prediction",
    {
      session_id: SESSION,
      sequence: seq,
      event_type: eventType,
      detector_mode: "anti_transfer",
      model_version: "turn-detector-at-0.1.0",
      event_end_ms: seq * 10,
      inference_ms: 5.1,
      confidence,
      decision_policy_version: "policy-0.1",
      reason_codes: ["stable_pause"],
    },
    seq,
  );

export const decision = (d: string, seq: number, gen: number | null) =>
  wire(
    "turn_decision",
    {
      decision: d,
      source_event_sequence: 3,
      last_user_frame_ms: 2000,
      committed_ms: 2300,
      confidence: 0.88,
      reason_codes: ["consecutive_frames_met"],
    },
    seq,
    { gen },
  );

export const transcript = (seq: number, gen: number | null, revision = 1, text = "hello there", isFinal = false) =>
  wire(
    "transcript",
    { segment_id: `seg_a${gen ?? 0}`, role: "assistant", text, is_final: isFinal, revision, start_ms: 100, end_ms: null },
    seq,
    { gen },
  );

export const hello = (seq = 1) =>
  wire(
    "hello",
    {
      protocol_version: 1,
      server_version: "0.1.0",
      session: {
        session_id: SESSION,
        trial_id: "tri_test0001",
        transport: "simulation",
        trial_type: "simulated",
        scenario_id: "pause_then_continue",
        trial_label: null,
        effective_config: effectiveConfigWire(),
        started_at: "2026-10-22T08:00:00Z",
      },
    },
    seq,
  );

export const createSessionResponseWire = () => ({
  session_id: SESSION,
  trial_id: "tri_test0001",
  effective_config: effectiveConfigWire(),
  events_url: `/ws/sessions/${SESSION}`,
  livekit: null,
});
