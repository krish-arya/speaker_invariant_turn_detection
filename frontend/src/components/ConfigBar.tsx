import { useId, useState } from "react";
import type { DetectorMode, Transport } from "../api/schemas";
import { useApp } from "../app/context";
import type { PolicyDraft } from "../lib/policy";
import { DETECTOR_LABELS } from "../lib/labels";
import { useSession } from "../state/sessionStore";
import { isSessionBusy } from "../state/sessionReducer";
import { Icon } from "./Icon";
import { PolicyFields, policyDraftErrors } from "./PolicyFields";

export interface ConsoleDraft {
  detectorMode: DetectorMode | null;
  scenarioId: string | null;
  transport: Transport;
  policy: PolicyDraft;
  seed: string;
  trialLabel: string;
  speculativePrefill: boolean;
}

/** Detector-mode and scenario bar (frontend.md §6.2). Frozen while a session is active. */
export function ConfigBar({ draft, onChange }: { draft: ConsoleDraft; onChange(next: ConsoleDraft): void }) {
  const { caps, mock } = useApp();
  const busy = useSession(isSessionBusy);
  const effective = useSession((s) => s.effectiveConfig);
  const phase = useSession((s) => s.phase);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const ids = useId();

  const scenarios = caps.scenarios.filter((s) => s.transports.includes(draft.transport));
  const scenario = scenarios.find((s) => s.id === draft.scenarioId) ?? null;
  const detector = caps.detectors.find((d) => d.mode === draft.detectorMode) ?? null;
  const errors = policyDraftErrors(draft.policy, caps.policyLimits);

  const setTransport = (t: Transport) => {
    const first = caps.scenarios.find((s) => s.transports.includes(t));
    const keep = caps.scenarios.find((s) => s.id === draft.scenarioId && s.transports.includes(t));
    onChange({ ...draft, transport: t, scenarioId: keep?.id ?? first?.id ?? null });
  };

  const transportOptions: Array<{ id: Transport; label: string; available: boolean; reason: string | null }> = [
    {
      id: "simulation",
      label: "Simulation (scripted audio)",
      available: caps.transports.simulation.available,
      reason: caps.transports.simulation.reason,
    },
    {
      id: "livekit",
      label: "Live microphone (LiveKit)",
      available: caps.transports.livekit.available && !mock,
      reason: mock ? "Not available in mock mode" : caps.transports.livekit.reason,
    },
  ];

  return (
    <section className="config-bar" aria-label="Session configuration">
      <div className="config-row">
        <div className="field">
          <label htmlFor={`${ids}-det`}>Detector</label>
          <select
            id={`${ids}-det`}
            value={draft.detectorMode ?? ""}
            disabled={busy}
            onChange={(e) => onChange({ ...draft, detectorMode: (e.target.value || null) as DetectorMode | null })}
            aria-describedby={`${ids}-det-help`}
          >
            {caps.detectors.length === 0 && <option value="">No detectors registered</option>}
            {caps.detectors.map((d) => (
              <option key={d.mode} value={d.mode} disabled={!d.available}>
                {d.label || DETECTOR_LABELS[d.mode]}
                {!d.available ? " — unavailable" : ""}
              </option>
            ))}
          </select>
          <p className="field-help" id={`${ids}-det-help`}>
            {detector ? (
              <>
                <span className="mono">{detector.modelVersion}</span>
                {detector.probabilistic ? " · probabilistic (confidence shown)" : " · deterministic (no confidence)"}
              </>
            ) : (
              "Choose a detector"
            )}
            {caps.detectors
              .filter((d) => !d.available)
              .map((d) => (
                <span key={d.mode} className="unavailable-note">
                  {" "}
                  {d.label}: {d.unavailableReason ?? "unavailable"}.
                </span>
              ))}
          </p>
        </div>

        <fieldset className="field transport-field" disabled={busy}>
          <legend>Transport</legend>
          <div className="segmented">
            {transportOptions.map((t) => (
              <label key={t.id} className={`segment${draft.transport === t.id ? " is-selected" : ""}${!t.available ? " is-disabled" : ""}`} title={!t.available ? t.reason ?? "Unavailable" : undefined}>
                <input
                  type="radio"
                  name={`${ids}-transport`}
                  value={t.id}
                  checked={draft.transport === t.id}
                  disabled={!t.available}
                  onChange={() => setTransport(t.id)}
                />
                <span>{t.label}</span>
              </label>
            ))}
          </div>
          {!transportOptions[1]!.available && (
            <p className="field-help">Live microphone: {transportOptions[1]!.reason ?? "unavailable"}.</p>
          )}
        </fieldset>

        <div className="field">
          <label htmlFor={`${ids}-scn`}>Scenario</label>
          <select
            id={`${ids}-scn`}
            value={draft.scenarioId ?? ""}
            disabled={busy || scenarios.length === 0}
            onChange={(e) => onChange({ ...draft, scenarioId: e.target.value || null })}
            aria-describedby={`${ids}-scn-help`}
          >
            {scenarios.length === 0 && <option value="">No scenarios for this transport</option>}
            {scenarios.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
          <p className="field-help" id={`${ids}-scn-help`}>
            {scenario
              ? `${scenario.description} ${scenario.labelled ? "Labelled — task metrics available." : "Unlabelled — latency and events only."}`
              : "Only scenarios the server supports are listed."}
          </p>
        </div>
      </div>

      <div className="config-footer">
        <button
          type="button"
          className="btn btn-ghost"
          aria-expanded={advancedOpen}
          aria-controls={`${ids}-adv`}
          onClick={() => setAdvancedOpen((o) => !o)}
        >
          <Icon name="chevron" size={14} />
          Advanced configuration
          {Object.keys(draft.policy).length > 0 && <span className="pill">custom thresholds</span>}
        </button>
        {busy ? (
          <p className="frozen-note" role="status">
            Configuration is frozen while a session is active. End the session to change it.
          </p>
        ) : phase === "ended" && effective ? (
          <p className="frozen-note">Changes apply to the next session.</p>
        ) : null}
        {errors.length > 0 && (
          <p className="field-error" role="alert">
            {errors.length} invalid threshold{errors.length > 1 ? "s" : ""} — fix before starting.
          </p>
        )}
      </div>

      {advancedOpen && (
        <div className="advanced-drawer" id={`${ids}-adv`}>
          <p className="drawer-intro">
            Thresholds are experimental parameters recorded verbatim in each trial. Server default policy{" "}
            <span className="mono">{caps.policyDefaults.version}</span>. Only changed fields are sent as overrides; the trial
            records the server-accepted values.
          </p>
          <PolicyFields
            defaults={caps.policyDefaults}
            limits={caps.policyLimits}
            draft={draft.policy}
            onChange={(policy) => onChange({ ...draft, policy })}
            disabled={busy}
          />
          <div className="config-row">
            <div className="field">
              <label htmlFor={`${ids}-seed`}>Seed (optional)</label>
              <input
                id={`${ids}-seed`}
                type="number"
                inputMode="numeric"
                value={draft.seed}
                disabled={busy}
                onChange={(e) => onChange({ ...draft, seed: e.target.value })}
              />
            </div>
            <div className="field">
              <label htmlFor={`${ids}-label`}>Trial label (optional)</label>
              <input
                id={`${ids}-label`}
                type="text"
                maxLength={120}
                value={draft.trialLabel}
                disabled={busy}
                onChange={(e) => onChange({ ...draft, trialLabel: e.target.value })}
              />
            </div>
            <div className="field">
              <label className="check" htmlFor={`${ids}-spec`}>
                <input
                  id={`${ids}-spec`}
                  type="checkbox"
                  checked={draft.speculativePrefill}
                  disabled={busy}
                  onChange={(e) => onChange({ ...draft, speculativePrefill: e.target.checked })}
                />
                Speculative prefill
              </label>
              <p className="field-help">
                Starts the LLM during a possible end on a stable partial transcript; the result is reused only if the transcript is
                unchanged when the turn is committed. Recorded in the trial configuration.
              </p>
            </div>
            <div className="field field-actions">
              <button type="button" className="btn btn-ghost" disabled={busy} onClick={() => onChange({ ...draft, policy: {} })}>
                Reset thresholds to server defaults
              </button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
