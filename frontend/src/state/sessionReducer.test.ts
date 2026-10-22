import { describe, expect, it } from "vitest";
import { createSessionResponseSchema, parseEnvelope } from "../api/schemas";
import { createSessionResponseWire, decision, hello, prediction, SESSION, status, transcript, wire } from "../test/fixtures";
import {
  canInterrupt,
  initialSessionState,
  isSessionBusy,
  selectDisplayPhase,
  sessionReducer,
  TIMELINE_LIMIT,
  type SessionAction,
  type SessionViewState,
} from "./sessionReducer";

function run(state: SessionViewState, ...actions: SessionAction[]): SessionViewState {
  return actions.reduce(sessionReducer, state);
}

const feed = (...msgs: unknown[]): SessionAction => ({ type: "envelopes", results: msgs.map((m) => parseEnvelope(m)) });

function started(): SessionViewState {
  return run(
    initialSessionState(),
    { type: "start_requested", transport: "simulation", isMock: false, needsPermission: false },
    { type: "session_created", response: createSessionResponseSchema.parse(createSessionResponseWire()) },
    { type: "transport_status", status: "connected" },
  );
}

describe("session state model", () => {
  it("starts idle and moves to connecting on start", () => {
    const s0 = initialSessionState();
    expect(s0.phase).toBe("idle");
    const s1 = sessionReducer(s0, { type: "start_requested", transport: "simulation", isMock: false, needsPermission: false });
    expect(s1.phase).toBe("connecting");
    const s2 = sessionReducer(s0, { type: "start_requested", transport: "livekit", isMock: false, needsPermission: true });
    expect(s2.phase).toBe("requesting_permission");
    expect(sessionReducer(s2, { type: "permission_granted" }).phase).toBe("connecting");
  });

  it("derives phase only from session_status envelopes", () => {
    let s = run(started(), feed(hello(1), status("listening", "connecting", 2)));
    expect(s.phase).toBe("listening");
    expect(s.effectiveConfig?.detectorMode).toBe("anti_transfer");
    // A prediction alone must not change the phase.
    s = run(s, feed(prediction("TURN_COMPLETE", 0.9, 3)));
    expect(s.phase).toBe("listening");
    expect(s.prediction?.eventType).toBe("TURN_COMPLETE");
    s = run(s, feed(status("user_speaking", "listening", 4), status("possible_end", "user_speaking", 5)));
    expect(s.phase).toBe("possible_end");
    s = run(s, feed(decision("TURN_COMPLETE", 6, 1), status("generating", "possible_end", 7, 1)));
    expect(s.phase).toBe("generating");
    expect(s.currentGeneration).toBe(1);
    expect(canInterrupt(s)).toBe(true);
  });

  it("keeps predictions and committed decisions distinct in the timeline", () => {
    const s = run(started(), feed(prediction("POSSIBLE_END", 0.72, 1), decision("TURN_COMPLETE", 2, 1)));
    const [p, d] = s.timeline;
    expect(p?.emphasis).toBe("prediction");
    expect(p?.label).toContain("Predicted");
    expect(p?.confidence).toBe(0.72);
    expect(d?.emphasis).toBe("decision");
    expect(d?.label).toContain("Committed");
    expect(d?.code).toBe("TURN_COMPLETE");
  });

  it("inserts a gap marker when sequence numbers jump", () => {
    const s = run(started(), feed(status("listening", "connecting", 1), status("user_speaking", "listening", 5)));
    expect(s.diagnostics.gaps).toBe(1);
    expect(s.diagnostics.missingMessages).toBe(3);
    const gap = s.timeline.find((i) => i.type === "gap");
    expect(gap?.label).toBe("3 messages missing");
    expect(gap?.detail).toContain("2–4");
    expect(s.phase).toBe("user_speaking");
  });

  it("drops duplicates and messages for other sessions", () => {
    const s = run(
      started(),
      feed(status("listening", "connecting", 1), status("listening", "connecting", 1), wire("vad", { speaking: true, probability: null, frame_end_ms: 1 }, 2, { session: "ses_other" })),
    );
    expect(s.diagnostics.duplicates).toBe(1);
    expect(s.diagnostics.foreign).toBe(1);
    expect(s.timeline).toHaveLength(1);
  });

  it("shows late (out-of-order) messages without rolling state back", () => {
    const s = run(started(), feed(status("listening", "connecting", 1), status("user_speaking", "listening", 3), status("listening", "connecting", 2)));
    expect(s.phase).toBe("user_speaking");
    expect(s.diagnostics.outOfOrder).toBe(1);
    expect(s.diagnostics.gaps).toBe(1);
    expect(s.timeline.some((i) => i.late)).toBe(true);
  });

  it("ignores transcript and playback start for a cancelled generation after BARGE_IN", () => {
    let s = run(
      started(),
      feed(decision("TURN_COMPLETE", 1, 2), status("agent_speaking", "generating", 2, 2), transcript(3, 2, 1, "Sure, the weather")),
    );
    expect(s.transcript).toHaveLength(1);
    s = run(s, feed(decision("BARGE_IN", 4, 2)));
    expect(s.cancelledGenerations).toContain(2);
    expect(s.stopping).toBe(true);
    s = run(
      s,
      feed(
        transcript(5, 2, 2, "late chunk"),
        wire("playback", { action: "start", reason: null }, 6, { gen: 2 }),
        wire("playback", { action: "flush", reason: "barge-in" }, 7, { gen: 2 }),
      ),
    );
    expect(s.diagnostics.staleDropped).toBe(2);
    expect(s.transcript[0]?.text).toBe("Sure, the weather");
    expect(s.transcript[0]?.interrupted).toBe(true); // never shown as delivered
    expect(s.stopping).toBe(false); // flush acknowledges the cancellation
    // A new generation is unaffected.
    s = run(s, feed(transcript(8, 3, 1, "new answer")));
    expect(s.transcript).toHaveLength(2);
  });

  it("updates transcript segments in place by segment id and revision", () => {
    let s = run(started(), feed(transcript(1, 1, 1, "Sure"), transcript(2, 1, 2, "Sure, the")));
    expect(s.transcript).toHaveLength(1);
    expect(s.transcript[0]?.text).toBe("Sure, the");
    s = run(s, feed(transcript(3, 1, 1, "stale revision")));
    expect(s.transcript[0]?.text).toBe("Sure, the");
  });

  it("bounds the timeline to the last 500 items", () => {
    const msgs = Array.from({ length: TIMELINE_LIMIT + 120 }, (_, i) =>
      wire("vad", { speaking: i % 2 === 0, probability: null, frame_end_ms: i }, i + 1),
    );
    const s = run(started(), feed(...msgs));
    expect(s.timeline).toHaveLength(TIMELINE_LIMIT);
    expect(s.timelineTrimmed).toBe(120);
    expect(s.timeline[s.timeline.length - 1]?.seq).toBe(TIMELINE_LIMIT + 120);
  });

  it("counts unknown kinds and invalid payloads in diagnostics", () => {
    const s = run(started(), feed(wire("mystery", {}, 1), wire("vad", { speaking: 1 }, 2), "garbage"));
    expect(s.diagnostics.unknownTotal).toBe(1);
    expect(s.diagnostics.unknownKinds.mystery).toBe(1);
    expect(s.diagnostics.invalid).toBe(2);
    expect(s.phase).toBe("connecting");
  });

  it("shows reconnecting from transport without inventing server state, and marks the gap", () => {
    let s = run(started(), feed(status("listening", "connecting", 1)), { type: "transport_status", status: "reconnecting" });
    expect(selectDisplayPhase(s)).toBe("reconnecting");
    expect(s.phase).toBe("listening");
    expect(s.timeline.some((i) => i.label.startsWith("Connection lost"))).toBe(true);
    s = sessionReducer(s, { type: "transport_status", status: "connected" });
    expect(selectDisplayPhase(s)).toBe("listening");
    expect(s.timeline.some((i) => i.label === "Connection restored")).toBe(true);
  });

  it("moves to error with a retryable message when permission is denied", () => {
    const s = run(initialSessionState(), { type: "start_requested", transport: "livekit", isMock: false, needsPermission: true }, {
      type: "local_error",
      error: { kind: "permission_denied", message: "Microphone access was blocked.", retryable: true, fatal: true },
    });
    expect(s.phase).toBe("error");
    expect(s.localError?.kind).toBe("permission_denied");
    expect(isSessionBusy(s)).toBe(false);
    expect(sessionReducer(s, { type: "dismiss_error" }).phase).toBe("idle");
  });

  it("ends cleanly: ending → ended keeps the summary but resets live-only state", () => {
    let s = run(started(), feed(status("listening", "connecting", 1), wire("audio_level", { source: "user", rms_dbfs: -20 }, 2)));
    expect(s.audioLevel.user).toBe(-20);
    expect(isSessionBusy(s)).toBe(true);
    s = sessionReducer(s, { type: "end_requested" });
    expect(s.phase).toBe("ending");
    // Late server state while ending does not flip the UI back.
    s = run(s, feed(status("listening", "user_speaking", 3)));
    expect(s.phase).toBe("ending");
    s = run(s, feed(status("ended", "ending", 4)), { type: "source_closed" });
    expect(s.phase).toBe("ended");
    expect(s.audioLevel.user).toBeNull();
    expect(isSessionBusy(s)).toBe(false);
    expect(s.sessionId).toBe(SESSION);
    expect(sessionReducer(s, { type: "clear_view" }).sessionId).toBeNull();
  });

  it("starting a new session discards the previous transcript", () => {
    let s = run(started(), feed(transcript(1, 1, 1, "secret")));
    s = sessionReducer(s, { type: "start_requested", transport: "simulation", isMock: false, needsPermission: false });
    expect(s.transcript).toHaveLength(0);
    expect(s.timeline).toHaveLength(0);
  });
});
