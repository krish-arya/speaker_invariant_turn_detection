import type { PolicyConfig } from "../api/schemas";

export type PolicyNumericKey = Exclude<keyof PolicyConfig, "version">;

export interface PolicyField {
  key: PolicyNumericKey;
  /** snake_case name used on the wire, in policy_limits and policy_overrides */
  wire: string;
  label: string;
  unit: "ms" | "prob" | "frames";
  step: number;
  help: string;
  /** Which detectors the field affects; shown so users don't tune irrelevant knobs. */
  appliesTo: "baseline" | "learned" | "all";
}

export const POLICY_FIELDS: PolicyField[] = [
  {
    key: "silenceTimeoutMs",
    wire: "silence_timeout_ms",
    label: "Silence timeout",
    unit: "ms",
    step: 10,
    help: "Baseline: silence after speech before TURN_COMPLETE is committed.",
    appliesTo: "baseline",
  },
  {
    key: "possibleEndMs",
    wire: "possible_end_ms",
    label: "Possible-end onset",
    unit: "ms",
    step: 10,
    help: "Silence after speech before POSSIBLE_END is predicted.",
    appliesTo: "all",
  },
  {
    key: "completeThreshold",
    wire: "complete_threshold",
    label: "Complete threshold",
    unit: "prob",
    step: 0.01,
    help: "Learned detectors: minimum p(turn end) for a TURN_COMPLETE prediction.",
    appliesTo: "learned",
  },
  {
    key: "possibleEndThreshold",
    wire: "possible_end_threshold",
    label: "Possible-end threshold",
    unit: "prob",
    step: 0.01,
    help: "Learned detectors: minimum p(turn end) for a POSSIBLE_END prediction.",
    appliesTo: "learned",
  },
  {
    key: "consecutiveFrames",
    wire: "consecutive_frames",
    label: "Consecutive frames",
    unit: "frames",
    step: 1,
    help: "Consecutive TURN_COMPLETE predictions required before the orchestrator commits.",
    appliesTo: "all",
  },
  {
    key: "minSilenceGuardMs",
    wire: "min_silence_guard_ms",
    label: "Minimum silence guard",
    unit: "ms",
    step: 10,
    help: "No TURN_COMPLETE may be committed before this much silence (endpointing guard).",
    appliesTo: "all",
  },
  {
    key: "maxSilenceMs",
    wire: "max_silence_ms",
    label: "Maximum silence",
    unit: "ms",
    step: 50,
    help: "Learned detectors: commit anyway after this much silence (bounded worst case).",
    appliesTo: "learned",
  },
  {
    key: "bargeInMinSpeechMs",
    wire: "barge_in_min_speech_ms",
    label: "Barge-in minimum speech",
    unit: "ms",
    step: 10,
    help: "User speech during agent output must last this long to count as barge-in.",
    appliesTo: "all",
  },
  {
    key: "backchannelMaxMs",
    wire: "backchannel_max_ms",
    label: "Backchannel maximum",
    unit: "ms",
    step: 10,
    help: "Speech this short that ends during agent output is treated as a backchannel.",
    appliesTo: "all",
  },
  {
    key: "minSpeechMs",
    wire: "min_speech_ms",
    label: "Minimum speech",
    unit: "ms",
    step: 5,
    help: "Speech segments shorter than this are ignored while listening (clicks, coughs).",
    appliesTo: "all",
  },
];

export type PolicyDraft = Partial<Record<PolicyNumericKey, number>>;

/** Only fields that differ from the server defaults, keyed by wire name. */
export function toWireOverrides(draft: PolicyDraft, defaults: PolicyConfig): Record<string, number> | null {
  const out: Record<string, number> = {};
  for (const f of POLICY_FIELDS) {
    const v = draft[f.key];
    if (typeof v === "number" && Number.isFinite(v) && v !== defaults[f.key]) out[f.wire] = v;
  }
  return Object.keys(out).length ? out : null;
}

export function clampToLimits(
  value: number,
  field: PolicyField,
  limits: Record<string, [number, number]>,
): number {
  const lim = limits[field.wire];
  if (!lim) return value;
  return Math.min(lim[1], Math.max(lim[0], value));
}

export function validatePolicyValue(
  value: number,
  field: PolicyField,
  limits: Record<string, [number, number]>,
): string | null {
  if (!Number.isFinite(value)) return "Enter a number.";
  const lim = limits[field.wire];
  if (lim && (value < lim[0] || value > lim[1])) return `Must be between ${lim[0]} and ${lim[1]}.`;
  if (field.unit === "prob" && (value < 0 || value > 1)) return "Must be between 0 and 1.";
  if (field.unit === "frames" && !Number.isInteger(value)) return "Must be a whole number.";
  if (value < 0) return "Must not be negative.";
  return null;
}
