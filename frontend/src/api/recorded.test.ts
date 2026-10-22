/**
 * Real backend output (recorded by backend/scripts/record_frontend_fixture.py) must pass the
 * frontend's validators and fold cleanly through the session reducer. Re-record the fixture
 * whenever the wire contract changes.
 */
import { describe, expect, it } from "vitest";

import recorded from "../test/fixtures/backend-recorded.json";
import { initialSessionState, sessionReducer, type SessionViewState } from "../state/sessionReducer";
import {
  capabilitiesSchema,
  comparisonResultSchema,
  createSessionResponseSchema,
  evaluationStatusSchema,
  healthSchema,
  parseEnvelope,
  trialDetailSchema,
  trialListSchema,
} from "./schemas";

type RecordedSession = (typeof recorded.sessions)[number];

function fold(session: RecordedSession): SessionViewState {
  let s = sessionReducer(initialSessionState(), {
    type: "start_requested",
    transport: "simulation",
    isMock: false,
    needsPermission: false,
  });
  s = sessionReducer(s, {
    type: "session_created",
    response: createSessionResponseSchema.parse(session.create_response),
  });
  s = sessionReducer(s, { type: "transport_status", status: "connected" });
  return sessionReducer(s, { type: "envelopes", results: session.messages.map((m) => parseEnvelope(m)) });
}

describe("recorded backend output", () => {
  it.each(recorded.sessions.map((s) => [s.request.scenario_id, s] as const))(
    "every %s envelope validates",
    (_, session) => {
      const failures = session.messages
        .map((m) => parseEnvelope(m))
        .filter((r) => !r.ok)
        .map((r) => (r.ok ? "" : `${r.kind}: ${r.message}`));
      expect(failures).toEqual([]);
      expect(session.messages.length).toBeGreaterThan(100);
    },
  );

  it("folds a barge-in session without gaps, duplicates or unknown kinds", () => {
    const s = fold(recorded.sessions[0]!);
    expect(s.diagnostics.invalid).toBe(0);
    expect(s.diagnostics.unknownTotal).toBe(0);
    expect(s.diagnostics.gaps).toBe(0);
    expect(s.diagnostics.duplicates).toBe(0);
    expect(s.serverState).toBe("ended");
    expect(s.trialSummary?.labelled).toBe(true);
    expect(s.cancelledGenerations).toContain(1);
    expect(s.transcript.some((t) => t.role === "user")).toBe(true);
    expect(s.transcript.some((t) => t.role === "assistant")).toBe(true);
  });

  it("shows no transcript text when the session disabled it", () => {
    const s = fold(recorded.sessions[1]!);
    expect(s.transcript).toEqual([]);
    expect(s.effectiveConfig?.speculativePrefill).toBe(true);
    expect(s.serverState).toBe("ended");
  });

  it("parses every REST response the console consumes", () => {
    const rest = recorded.rest;
    expect(() => healthSchema.parse(rest.health)).not.toThrow();
    expect(() => capabilitiesSchema.parse(rest.capabilities)).not.toThrow();
    expect(() => evaluationStatusSchema.parse(rest.evaluation_started)).not.toThrow();
    expect(evaluationStatusSchema.parse(rest.evaluation_status).status).toBe("completed");
    expect(trialListSchema.parse(rest.trial_list).items.length).toBe(4);
    expect(trialDetailSchema.parse(rest.trial_detail).events.length).toBeGreaterThan(0);
    expect(comparisonResultSchema.parse(rest.comparison).groups.length).toBe(2);
  });
});
