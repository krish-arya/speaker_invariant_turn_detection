import { useId } from "react";
import type { PolicyConfig } from "../api/schemas";
import { POLICY_FIELDS, validatePolicyValue, type PolicyDraft } from "../lib/policy";

const UNIT_LABEL = { ms: "ms", prob: "p", frames: "frames" } as const;
const APPLIES = { baseline: "Baseline only", learned: "Learned detectors", all: "All detectors" } as const;

export function policyDraftErrors(draft: PolicyDraft, limits: Record<string, [number, number]>): string[] {
  const errs: string[] = [];
  for (const f of POLICY_FIELDS) {
    const v = draft[f.key];
    if (v === undefined) continue;
    const e = validatePolicyValue(v, f, limits);
    if (e) errs.push(`${f.label}: ${e}`);
  }
  return errs;
}

/** Advanced endpointing thresholds, bounded by the server's policy_limits. */
export function PolicyFields({
  defaults,
  limits,
  draft,
  onChange,
  disabled,
}: {
  defaults: PolicyConfig;
  limits: Record<string, [number, number]>;
  draft: PolicyDraft;
  onChange(next: PolicyDraft): void;
  disabled: boolean;
}) {
  const base = useId();
  return (
    <div className="policy-grid">
      {POLICY_FIELDS.map((f) => {
        const id = `${base}-${f.key}`;
        const value = draft[f.key] ?? defaults[f.key];
        const lim = limits[f.wire];
        const err = validatePolicyValue(value, f, limits);
        const changed = draft[f.key] !== undefined && draft[f.key] !== defaults[f.key];
        return (
          <div className={`field${err ? " has-error" : ""}`} key={f.key}>
            <label htmlFor={id}>
              {f.label}
              {changed && <span className="changed-dot" title="Differs from server default"> • changed</span>}
            </label>
            <div className="input-with-unit">
              <input
                id={id}
                type="number"
                inputMode="decimal"
                value={Number.isFinite(value) ? value : ""}
                min={lim?.[0]}
                max={lim?.[1]}
                step={f.step}
                disabled={disabled}
                aria-invalid={err ? true : undefined}
                aria-describedby={`${id}-help${err ? ` ${id}-err` : ""}`}
                onChange={(e) => {
                  const raw = e.target.value;
                  const next = { ...draft };
                  if (raw === "") delete next[f.key];
                  else next[f.key] = Number(raw);
                  onChange(next);
                }}
              />
              <span className="unit" aria-hidden="true">
                {UNIT_LABEL[f.unit]}
              </span>
            </div>
            <p className="field-help" id={`${id}-help`}>
              {f.help} <span className="applies">{APPLIES[f.appliesTo]}</span>
              {lim && (
                <span className="range">
                  {" "}
                  Range {lim[0]}–{lim[1]}; default {defaults[f.key]}.
                </span>
              )}
            </p>
            {err && (
              <p className="field-error" id={`${id}-err`}>
                {err}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
