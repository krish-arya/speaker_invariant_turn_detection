/**
 * Contract drift guard: every field and enum value in the backend's wire models must be
 * known to the frontend validators / request builders. Skipped when the backend source
 * is not checked out next to the frontend.
 */
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

const root = resolve(__dirname, "../../..");
const eventsPy = resolve(root, "backend/src/turnsense/events.py");
const schemasTs = readFileSync(resolve(__dirname, "schemas.ts"), "utf-8");
const clientTs = readFileSync(resolve(__dirname, "client.ts"), "utf-8");

/** class name -> field names (annotated assignments at class-body indentation). */
function pythonModels(src: string): Map<string, { fields: string[]; enumValues: string[] }> {
  const out = new Map<string, { fields: string[]; enumValues: string[] }>();
  let current: string | null = null;
  for (const line of src.split(/\r?\n/)) {
    const cls = /^class (\w+)\(/.exec(line);
    if (cls) {
      current = cls[1]!;
      out.set(current, { fields: [], enumValues: [] });
      continue;
    }
    if (/^\S/.test(line)) current = null;
    if (!current) continue;
    const field = /^ {4}([a-z_][a-z0-9_]*)\s*:/.exec(line);
    if (field && field[1] !== "model_config") out.get(current)!.fields.push(field[1]!);
    const enumVal = /^ {4}[A-Z_][A-Z0-9_]*\s*=\s*"([^"]+)"/.exec(line);
    if (enumVal) out.get(current)!.enumValues.push(enumVal[1]!);
  }
  return out;
}

const VALIDATED = [
  "PolicyConfig",
  "VadConfig",
  "ProviderInfo",
  "EffectiveConfig",
  "SessionInfo",
  "HelloPayload",
  "SessionStatusPayload",
  "VadPayload",
  "TurnEvent",
  "TurnDecisionPayload",
  "TranscriptPayload",
  "TraceEvent",
  "LatencyPayload",
  "PercentileSummary",
  "AudioLevelPayload",
  "PlaybackPayload",
  "ErrorPayload",
  "TrialMetrics",
  "TrialCounts",
  "TrialSummaryPayload",
  "Envelope",
  "LiveKitJoinInfo",
  "CreateSessionResponse",
  "EvaluationStatus",
];
const REQUESTS = ["CreateSessionRequest", "CreateEvaluationRequest"];

describe.skipIf(!existsSync(eventsPy))("wire contract (backend events.py)", () => {
  const models = pythonModels(existsSync(eventsPy) ? readFileSync(eventsPy, "utf-8") : "");

  it.each(VALIDATED)("frontend validates every field of %s", (name) => {
    const m = models.get(name);
    expect(m, `${name} not found in events.py`).toBeDefined();
    const missing = m!.fields.filter((f) => !new RegExp(`\\b${f}\\s*:`).test(schemasTs));
    expect(missing).toEqual([]);
  });

  it.each(REQUESTS)("client sends only known fields of %s", (name) => {
    const m = models.get(name);
    expect(m).toBeDefined();
    const missing = m!.fields.filter((f) => !new RegExp(`\\b${f}\\s*:`).test(clientTs));
    // client_build is sent for sessions only; everything else must be wired.
    expect(missing).toEqual([]);
  });

  it("knows every backend enum value", () => {
    const enums = [...models.entries()].filter(([, m]) => m.enumValues.length > 0);
    expect(enums.length).toBeGreaterThan(5);
    const missing = enums.flatMap(([n, m]) => m.enumValues.filter((v) => !schemasTs.includes(`"${v}"`)).map((v) => `${n}.${v}`));
    expect(missing).toEqual([]);
  });
});
